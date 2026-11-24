"""Kiểm thử đơn vị cho thuật toán SGD và IOptimizer (Chỉ tiêu T13, Quy tắc R7)."""

import numpy as np
import pytest

from nnlib.exceptions import InvalidConfigError
from nnlib.optim.sgd import SGD
from nnlib.tensor import Parameter
from tests.fakes.fake_layer import FakeLayer


def test_sgd_no_momentum_step_t13() -> None:
    """Kiểm thử T13: Khớp giá trị tính tay qua 2 bước khi không dùng momentum."""
    param = Parameter(np.array([1.0], dtype=np.float32), name="w")
    sgd = SGD(lr=0.1, momentum=0.0)

    # Bước 1: theta_0 = 1.0, g_0 = 0.5 -> theta_1 = 1.0 - 0.1 * 0.5 = 0.95
    param.grad = np.array([0.5], dtype=np.float32)
    sgd.step([param])
    assert np.allclose(param.data, [0.95])

    # Bước 2: theta_1 = 0.95, g_1 = 0.2 -> theta_2 = 0.95 - 0.1 * 0.2 = 0.93
    param.grad = np.array([0.2], dtype=np.float32)
    sgd.step([param])
    assert np.allclose(param.data, [0.93])


def test_sgd_with_momentum_step_t13() -> None:
    """Kiểm thử T13: Khớp giá trị tính tay qua 2 bước khi dùng momentum mu=0.9."""
    param = Parameter(np.array([1.0], dtype=np.float32), name="w")
    sgd = SGD(lr=0.1, momentum=0.9)

    # Bước 1: v_1 = 0.9 * 0 - 0.1 * 0.5 = -0.05
    #         theta_1 = 1.0 + (-0.05) = 0.95
    param.grad = np.array([0.5], dtype=np.float32)
    sgd.step([param])
    assert np.allclose(param.data, [0.95])

    # Bước 2: v_2 = 0.9 * (-0.05) - 0.1 * 0.2 = -0.045 - 0.02 = -0.065
    #         theta_2 = 0.95 + (-0.065) = 0.885
    param.grad = np.array([0.2], dtype=np.float32)
    sgd.step([param])
    assert np.allclose(param.data, [0.885])


@pytest.mark.parametrize("invalid_lr", [0.0, -0.1, True, float("nan"), float("inf")])
def test_sgd_invalid_lr_rule_r7(invalid_lr: object) -> None:
    """Rule R7: Learning rate phải là số thực dương > 0, loại trừ bool và nan."""
    with pytest.raises(InvalidConfigError):
        SGD(lr=invalid_lr)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "invalid_momentum", [-0.1, 1.0, 1.5, True, float("nan"), float("inf")]
)
def test_sgd_invalid_momentum_rule_r7(invalid_momentum: object) -> None:
    """Rule R7: Momentum phải thuộc [0, 1), loại trừ bool và nan."""
    with pytest.raises(InvalidConfigError):
        SGD(lr=0.01, momentum=invalid_momentum)  # type: ignore[arg-type]


def test_sgd_zero_grad() -> None:
    """zero_grad ủy quyền gọi param.zero_grad() cho tất cả tham số."""
    p1 = Parameter(np.ones((2, 2), dtype=np.float32), name="p1")
    p2 = Parameter(np.ones(2, dtype=np.float32), name="p2")
    p1.grad = np.full((2, 2), 2.5, dtype=np.float32)
    p2.grad = np.full(2, 1.5, dtype=np.float32)

    sgd = SGD(lr=0.1)
    sgd.zero_grad([p1, p2])

    assert np.all(p1.grad == 0.0)
    assert np.all(p2.grad == 0.0)


def test_sgd_with_fake_layer() -> None:
    """Kiểm thử tối ưu hóa trên FakeLayer độc lập."""
    layer = FakeLayer(in_features=2, out_features=1)
    params = layer.parameters()

    params[0].grad = np.ones((2, 1), dtype=np.float32)
    params[1].grad = np.ones(1, dtype=np.float32)

    sgd = SGD(lr=0.05, momentum=0.0)
    sgd.step(params)

    assert np.allclose(params[0].data, 0.95)
    assert np.allclose(params[1].data, -0.05)
