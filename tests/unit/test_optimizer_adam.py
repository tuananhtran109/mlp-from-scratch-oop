"""Kiểm thử đơn vị cho thuật toán Adam (Chỉ tiêu T14, Quy tắc R7)."""

from __future__ import annotations

import numpy as np
import pytest

from nnlib.exceptions import InvalidConfigError, ShapeMismatchError
from nnlib.layers.dense import Dense
from nnlib.optim.adam import Adam
from nnlib.optim.base import IOptimizer
from nnlib.tensor import Parameter


def test_adam_first_step_bias_correction_t14() -> None:
    """Kiểm thử T14: Tại t=1, nhờ bias correction nên delta_theta ~ -lr * sign(g).

    Phép tính tay tại bước t = 1 (lr = 0.001, beta1 = 0.9, beta2 = 0.999, eps = 1e-8):
        Với g = [0.5, -2.0]:
        - m_1 = 0.1 * g = [0.05, -0.2] -> m_hat_1 = m_1 / (1 - 0.9) = [0.5, -2.0] = g
        - v_1 = 0.001 * g^2 = [0.00025, 0.004] -> v_hat_1 = v_1 / (1 - 0.999) = g^2
        - sqrt(v_hat_1) = |g| = [0.5, 2.0]
        - delta_theta = -0.001 * g / (|g| + 1e-8) ~ [-0.001, +0.001] = -lr * sign(g)
    """
    theta_init = np.array([1.0, -3.0], dtype=np.float64)
    param = Parameter(theta_init.copy(), name="w", dtype=np.float64)
    grad = np.array([0.5, -2.0], dtype=np.float64)
    param.grad = grad

    adam = Adam(lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8)
    assert isinstance(adam, IOptimizer)
    adam.step([param])

    delta = param.data - theta_init
    expected_approx = -0.001 * np.sign(grad)
    # Sai số so với -lr * sign(g) chỉ do eps=1e-8 ở mẫu số (rất nhỏ ~ 2e-11)
    np.testing.assert_allclose(delta, expected_approx, rtol=1e-6, atol=1e-9)
    assert adam.t == 1


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_adam_two_steps_manual_calculation_t14(dtype: type[np.floating]) -> None:
    """Kiểm thử T14: Khớp giá trị tính tay qua 2 bước liên tiếp (sai số < 1e-6).

    Phép tính tay với theta_0 = 1.0, lr = 0.001, beta1 = 0.9, beta2 = 0.999, eps = 1e-8:
        Bước 1 (t = 1, g_1 = 0.5):
            m_1 = 0.9 * 0 + 0.1 * 0.5 = 0.05
            v_1 = 0.999 * 0 + 0.001 * 0.25 = 0.00025
            m_hat_1 = 0.05 / (1 - 0.9) = 0.5
            v_hat_1 = 0.00025 / (1 - 0.999) = 0.25
            theta_1 = 1.0 - 0.001 * 0.5 / (0.5 + 1e-8) = 0.9990000000199999

        Bước 2 (t = 2, g_2 = -0.2):
            m_2 = 0.9 * 0.05 + 0.1 * (-0.2) = 0.025
            v_2 = 0.999 * 0.00025 + 0.001 * 0.04 = 0.00028975
            m_hat_2 = 0.025 / (1 - 0.9^2) = 0.025 / 0.19 = 0.13157894736842105
            v_hat_2 = 0.00028975 / (1 - 0.999^2) = 0.00028975 / 0.001999
                    = 0.14494747373686843
            sqrt(v_hat_2) = 0.3807196786835012
            theta_2 = theta_1 - 0.001 * 0.13157894736842105
                      / (0.3807196786835012 + 1e-8)
                    = 0.9986543941276779
    """
    param = Parameter(np.array([1.0], dtype=dtype), name="w", dtype=dtype)
    adam = Adam(lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8)

    # Bước 1
    param.grad = np.array([0.5], dtype=dtype)
    adam.step([param])
    expected_theta_1 = np.array([0.9990000000199999], dtype=dtype)
    np.testing.assert_allclose(param.data, expected_theta_1, rtol=1e-6, atol=1e-7)

    # Bước 2
    param.grad = np.array([-0.2], dtype=dtype)
    adam.step([param])
    expected_theta_2 = np.array([0.9986543941276779], dtype=dtype)
    np.testing.assert_allclose(param.data, expected_theta_2, rtol=1e-6, atol=1e-7)
    assert adam.t == 2


@pytest.mark.parametrize(
    "invalid_lr",
    [0.0, -0.001, True, np.bool_(True), float("nan"), float("inf"), "0.001"],
)
def test_adam_invalid_lr_rule_r7(invalid_lr: object) -> None:
    """Rule R7: lr <= 0, NaN, inf hoặc bool phải ném InvalidConfigError."""
    with pytest.raises(InvalidConfigError):
        Adam(lr=invalid_lr)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "invalid_beta",
    [-0.1, 1.0, 1.2, True, np.bool_(False), float("nan"), float("inf"), "0.9"],
)
def test_adam_invalid_betas_rule_r7(invalid_beta: object) -> None:
    """Rule R7: beta1 và beta2 ngoài [0, 1) hoặc kiểu không hợp lệ ném lỗi."""
    with pytest.raises(InvalidConfigError):
        Adam(lr=0.001, beta1=invalid_beta)  # type: ignore[arg-type]

    with pytest.raises(InvalidConfigError):
        Adam(lr=0.001, beta2=invalid_beta)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "invalid_eps",
    [0.0, -1e-8, True, np.bool_(True), float("nan"), float("inf"), "1e-8"],
)
def test_adam_invalid_eps_rule_r7(invalid_eps: object) -> None:
    """Rule R7: eps <= 0, NaN, inf hoặc bool phải ném InvalidConfigError."""
    with pytest.raises(InvalidConfigError):
        Adam(lr=0.001, eps=invalid_eps)  # type: ignore[arg-type]


def test_adam_valid_boundary_beta_zero_rule_r7() -> None:
    """Rule R7 (biên hợp lệ): beta1=0.0 và beta2=0.0 nằm trong [0, 1) được chấp nhận."""
    param = Parameter(np.array([2.0], dtype=np.float64), name="w", dtype=np.float64)
    param.grad = np.array([4.0], dtype=np.float64)

    adam = Adam(lr=0.1, beta1=0.0, beta2=0.0, eps=1e-8)
    adam.step([param])

    # Với beta1=0, beta2=0: m_1 = g = 4.0, v_1 = g^2 = 16.0
    # theta_1 = 2.0 - 0.1 * 4.0 / (4.0 + 1e-8) ~ 1.9
    np.testing.assert_allclose(param.data, [1.9], rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_adam_state_preserves_param_dtype(dtype: type[np.floating]) -> None:
    """Trạng thái moment m, v của Adam giữ nguyên dtype với param.data."""
    dense = Dense(in_features=3, out_features=2, dtype=dtype)
    params = dense.parameters()
    for p in params:
        p.grad = np.ones_like(p.data) * 0.25

    adam = Adam(lr=0.001)
    adam.step(params)

    assert len(adam.m) == 2
    assert len(adam.v) == 2
    for idx, p in enumerate(params):
        assert adam.m[idx].dtype == p.data.dtype
        assert adam.v[idx].dtype == p.data.dtype
        assert p.data.dtype == np.dtype(dtype)


def test_adam_zero_grad_and_state_uses_index_not_id() -> None:
    """zero_grad gọi param.zero_grad(); state khóa theo index nguyên không dùng id()."""
    p1 = Parameter(np.ones((2, 2), dtype=np.float32), name="w")
    p2 = Parameter(np.zeros(2, dtype=np.float32), name="b")
    p1.grad = np.full((2, 2), 1.5, dtype=np.float32)
    p2.grad = np.full(2, -0.5, dtype=np.float32)

    adam = Adam(lr=0.01)
    adam.step([p1, p2])

    assert set(adam.m.keys()) == {0, 1}
    assert set(adam.v.keys()) == {0, 1}
    assert id(p1) not in adam.m
    assert id(p2) not in adam.m

    adam.zero_grad([p1, p2])
    assert np.all(p1.grad == 0.0)
    assert np.all(p2.grad == 0.0)


def test_adam_empty_params_and_mismatched_shape_guard() -> None:
    """Truyền params rỗng không tăng t; đổi shape tham số tại cùng vị trí ném lỗi."""
    adam = Adam(lr=0.001)
    adam.step([])
    assert adam.t == 0

    p1 = Parameter(np.ones((2, 2), dtype=np.float32), name="w1")
    p1.grad = np.ones((2, 2), dtype=np.float32)
    adam.step([p1])
    assert adam.t == 1

    p_wrong = Parameter(np.ones((3, 2), dtype=np.float32), name="w2")
    p_wrong.grad = np.ones((3, 2), dtype=np.float32)
    with pytest.raises(ShapeMismatchError):
        adam.step([p_wrong])
    assert adam.t == 1

    p_wrong_dtype = Parameter(
        np.ones((2, 2), dtype=np.float64), name="w3", dtype=np.float64
    )
    p_wrong_dtype.grad = np.ones((2, 2), dtype=np.float64)
    with pytest.raises(InvalidConfigError):
        adam.step([p_wrong_dtype])
    assert adam.t == 1
