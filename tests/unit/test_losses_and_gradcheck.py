"""Kiểm thử đơn vị cho Losses và GradientChecker (bao gồm T08, T09 và M2)."""

import math

import numpy as np
import pytest

from nnlib.exceptions import InvalidConfigError, ShapeMismatchError, StateError
from nnlib.gradcheck import GradientChecker
from nnlib.initializers import HeInitializer
from nnlib.layers import Dense, Dropout, Layer, ReLU, Sigmoid, Softmax, Tanh
from nnlib.losses import CrossEntropyLoss, ILossFunction, MSELoss
from nnlib.tensor import Tensor


def test_t08_cross_entropy_hand_value_zero_prob_and_gradcheck() -> None:
    """T08 (R4): Khớp -ln(0.7), p=0 không ra inf/NaN, gradient check < 1e-5."""
    ce = CrossEntropyLoss(eps=1e-12)
    assert isinstance(ce, ILossFunction)
    assert ce.eps == 1e-12

    # Ví dụ tính tay: 1 mẫu với xác suất lớp đúng = 0.7 -> loss = -ln(0.7)
    y_pred = Tensor([[0.2, 0.7, 0.1]], dtype=np.float64)
    y_true = Tensor([[0.0, 1.0, 0.0]], dtype=np.float64)
    loss = ce.forward(y_pred, y_true)
    assert math.isclose(loss, -math.log(0.7), rel_tol=1e-12)

    # p = 0 ở lớp đúng không ra inf hay NaN (R4)
    p_zero = Tensor([[0.0, 1.0, 0.0]], dtype=np.float32)
    y_zero = Tensor([[1.0, 0.0, 0.0]], dtype=np.float32)
    loss_zero = ce.forward(p_zero, y_zero)
    grad_zero = ce.backward()
    assert math.isfinite(loss_zero)
    assert grad_zero.dtype == np.float32
    assert not np.any(np.isnan(grad_zero.data))
    assert not np.any(np.isinf(grad_zero.data))

    # Gradient check bằng GradientChecker trên float64
    rng = np.random.default_rng(42)
    logits = rng.normal(0.0, 1.0, size=(5, 4)).astype(np.float64)
    probs = Softmax().forward(Tensor(logits, dtype=np.float64))
    targets = np.zeros((5, 4), dtype=np.float64)
    targets[np.arange(5), [0, 2, 1, 3, 2]] = 1.0

    checker = GradientChecker(epsilon=1e-5, tolerance=1e-5)
    err = checker.check_loss(ce, probs, Tensor(targets, dtype=np.float64))
    assert checker.is_valid(err)


def test_t09_backward_before_forward_raises_state_error() -> None:
    """T09 (R2): Gọi backward() trước forward() ở mọi Layer và Loss -> StateError."""
    dummy_grad = Tensor([[1.0, 2.0]], dtype=np.float32)
    layers: list[Layer] = [
        Dense(in_features=2, out_features=2),
        ReLU(),
        Sigmoid(),
        Tanh(),
        Softmax(),
        Dropout(rate=0.2, seed=0),
    ]
    for layer in layers:
        with pytest.raises(StateError):
            layer.backward(dummy_grad)

    losses: list[ILossFunction] = [MSELoss(), CrossEntropyLoss()]
    for loss_fn in losses:
        with pytest.raises(StateError):
            loss_fn.backward()


def test_mse_loss_hand_value_cache_isolation_and_gradcheck() -> None:
    """MSELoss khớp công thức tính tay, độc lập cache, giữ dtype và đạt gradcheck."""
    mse = MSELoss()
    raw_p = np.array([[1.0, 3.0], [2.0, -1.0]], dtype=np.float32)
    raw_y = np.array([[0.0, 1.0], [2.0, 1.0]], dtype=np.float32)

    # Hàng 0: ||[1, 2]||^2 = 5; Hàng 1: ||[0, -2]||^2 = 4 -> L = (5 + 4)/2 = 4.5
    loss = mse.forward(Tensor(raw_p), Tensor(raw_y))
    assert math.isclose(loss, 4.5, rel_tol=1e-6)

    # Sửa mảng gốc sau forward không làm sai gradient trong backward
    raw_p[:] = 999.0
    raw_y[:] = -999.0
    grad = mse.backward()
    assert grad.dtype == np.float32
    assert np.allclose(grad.data, [[1.0, 2.0], [0.0, -2.0]])

    # Gradient check trên float64
    rng = np.random.default_rng(7)
    p64 = Tensor(rng.normal(size=(4, 3)).astype(np.float64))
    y64 = Tensor(rng.normal(size=(4, 3)).astype(np.float64))
    checker = GradientChecker()
    assert checker.is_valid(checker.check_loss(mse, p64, y64))


def test_m2_gradient_checker_on_dense_and_all_activations() -> None:
    """M2: GradientChecker xác nhận Dense, ReLU, Sigmoid, Tanh, Softmax < 1e-5."""
    rng = np.random.default_rng(2026)
    checker = GradientChecker(epsilon=1e-5, tolerance=1e-5)
    mse = MSELoss()
    ce = CrossEntropyLoss()

    x_arr = rng.normal(0.0, 1.0, size=(4, 5)).astype(np.float64)
    x_arr = np.where(np.abs(x_arr) < 0.05, 0.25, x_arr)
    x = Tensor(x_arr, dtype=np.float64)
    y_reg = Tensor(rng.normal(0.0, 1.0, size=(4, 5)).astype(np.float64))

    # Kiểm tra tham số và đầu vào của Dense
    dense = Dense(
        in_features=5,
        out_features=3,
        weight_initializer=HeInitializer(seed=11),
        dtype=np.float64,
    )
    y_dense = Tensor(rng.normal(0.0, 1.0, size=(4, 3)).astype(np.float64))
    param_errors = checker.check_parameters(dense, mse, x, y_dense)
    assert set(param_errors.keys()) == {"0.weights", "1.bias"}
    assert checker.is_valid(param_errors)
    assert checker.is_valid(checker.check_input(dense, mse, x, y_dense))

    # Kiểm tra từng hàm kích hoạt với MSELoss / CrossEntropyLoss
    for act in (ReLU(), Sigmoid(), Tanh()):
        assert checker.is_valid(checker.check_input(act, mse, x, y_reg))

    y_onehot = np.zeros((4, 5), dtype=np.float64)
    y_onehot[np.arange(4), [0, 1, 2, 4]] = 1.0
    assert checker.is_valid(
        checker.check_input(Softmax(), ce, x, Tensor(y_onehot, dtype=np.float64))
    )


def test_losses_and_gradcheck_validation_errors() -> None:
    """Kiểm tra lỗi cấu hình, lỗi shape và yêu cầu float64 của GradientChecker."""
    with pytest.raises(TypeError):
        ILossFunction()  # type: ignore[abstract]

    for bad_eps in (0.0, -1e-6, 0.5, 1.0, float("nan"), True, "1e-7"):
        with pytest.raises(InvalidConfigError):
            CrossEntropyLoss(eps=bad_eps)  # type: ignore[arg-type]

    # Chấp nhận số thực NumPy cho eps, epsilon, tolerance
    assert CrossEntropyLoss(eps=np.float32(1e-6)).eps > 0.0
    gc_np = GradientChecker(epsilon=np.float64(1e-5), tolerance=np.float64(1e-4))
    assert gc_np.epsilon == 1e-5
    assert gc_np.tolerance == 1e-4

    with pytest.raises(InvalidConfigError):
        GradientChecker(epsilon=0.0)
    with pytest.raises(InvalidConfigError):
        GradientChecker(tolerance=-1e-5)
    with pytest.raises(InvalidConfigError):
        GradientChecker(epsilon=True)

    # Sai shape ở Loss
    with pytest.raises(ShapeMismatchError):
        CrossEntropyLoss().forward(
            Tensor([0.5, 0.5], dtype=np.float32),
            Tensor([1.0, 0.0], dtype=np.float32),
        )
    with pytest.raises(ShapeMismatchError):
        MSELoss().forward(
            Tensor([[1.0, 2.0]], dtype=np.float32),
            Tensor([[1.0, 2.0, 3.0]], dtype=np.float32),
        )

    # GradientChecker từ chối dữ liệu/tham số không phải float64
    checker = GradientChecker()
    with pytest.raises(InvalidConfigError):
        checker.check_loss(
            MSELoss(),
            Tensor([[1.0, 2.0]], dtype=np.float32),
            Tensor([[0.0, 1.0]], dtype=np.float32),
        )
    with pytest.raises(InvalidConfigError):
        checker.check_parameters(
            Dense(in_features=2, out_features=2, dtype=np.float32),
            MSELoss(),
            Tensor([[1.0, 2.0]], dtype=np.float64),
            Tensor([[0.0, 1.0]], dtype=np.float64),
        )
