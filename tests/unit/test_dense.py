"""Kiểm thử đơn vị cho lớp Dense (bao gồm T01, T02, T03)."""

import numpy as np
import pytest

from nnlib.exceptions import InvalidConfigError, ShapeMismatchError, StateError
from nnlib.initializers import HeInitializer, ZerosInitializer
from nnlib.layers import Dense
from nnlib.tensor import Tensor


def _relative_error(analytical: np.ndarray, numerical: np.ndarray) -> float:
    """Tính sai số tương đối cực đại theo đặc tả gradient check."""
    diff = np.abs(analytical - numerical)
    denom = np.maximum(np.abs(analytical) + np.abs(numerical), 1e-12)
    return float(np.max(diff / denom))


def test_t01_dense_forward_fixed_weights() -> None:
    """T01: Dense.forward với W, b cố định cho đúng shape và giá trị tính tay."""
    layer = Dense(in_features=2, out_features=3)
    layer.weights.data = np.array(
        [[1.0, 2.0, -1.0], [0.5, -1.0, 3.0]], dtype=np.float32
    )
    layer.bias.data = np.array([0.5, -0.5, 1.0], dtype=np.float32)

    x = Tensor([[2.0, 4.0], [-1.0, 2.0]], dtype=np.float32)
    y = layer.forward(x)

    # Hàng 0: [2*1 + 4*0.5 + 0.5, 2*2 + 4*(-1) - 0.5, 2*(-1) + 4*3 + 1.0]
    #       = [4.5, -0.5, 11.0]
    # Hàng 1: [-1*1 + 2*0.5 + 0.5, -1*2 + 2*(-1) - 0.5, -1*(-1) + 2*3 + 1.0]
    #       = [0.5, -4.5, 8.0]
    expected = np.array([[4.5, -0.5, 11.0], [0.5, -4.5, 8.0]], dtype=np.float32)

    assert y.shape == (2, 3)
    assert y.dtype == np.float32
    assert np.allclose(y.data, expected)


def test_t02_dense_invalid_shape_raises_error() -> None:
    """T02 (R1): Đầu vào không phải 2D (N, in_features) ném ShapeMismatchError."""
    layer = Dense(in_features=4, out_features=3)

    # Sai số chiều (1D và 3D)
    with pytest.raises(ShapeMismatchError):
        layer.forward(Tensor(np.ones((4,), dtype=np.float32)))
    with pytest.raises(ShapeMismatchError):
        layer.forward(Tensor(np.ones((2, 1, 4), dtype=np.float32)))

    # Sai số đặc trưng in_features
    with pytest.raises(ShapeMismatchError):
        layer.forward(Tensor(np.ones((2, 5), dtype=np.float32)))

    # Sai shape grad_out ở backward (sai out_features hoặc sai batch size)
    _ = layer.forward(Tensor(np.ones((2, 4), dtype=np.float32)))
    with pytest.raises(ShapeMismatchError):
        layer.backward(Tensor(np.ones((2, 5), dtype=np.float32)))
    with pytest.raises(ShapeMismatchError):
        layer.backward(Tensor(np.ones((3, 3), dtype=np.float32)))


def test_t03_dense_gradient_check_float64() -> None:
    """T03: Gradient check Dense (W, b, x) trên float64 có sai số < 1e-5."""
    rng = np.random.default_rng(123)
    layer = Dense(
        in_features=4,
        out_features=3,
        weight_initializer=HeInitializer(seed=42),
        bias_initializer=ZerosInitializer(),
        dtype=np.float64,
    )
    layer.bias.data = rng.normal(0.0, 0.5, size=(3,))

    x_arr = rng.normal(0.0, 1.0, size=(5, 4))
    u_arr = rng.normal(0.0, 1.0, size=(5, 3))
    eps = 1e-5

    def scalar_loss(inp: np.ndarray) -> float:
        out = layer.forward(Tensor(inp, dtype=np.float64)).data
        return float(np.sum(out * u_arr))

    # Gradient giải tích
    _ = layer.forward(Tensor(x_arr, dtype=np.float64))
    grad_x_ana = layer.backward(Tensor(u_arr, dtype=np.float64)).data
    grad_w_ana = layer.weights.grad.copy()
    grad_b_ana = layer.bias.grad.copy()

    # Gradient số học cho W
    grad_w_num = np.zeros_like(layer.weights.data)
    for idx in np.ndindex(layer.weights.shape):
        orig = layer.weights.data[idx]
        layer.weights.data[idx] = orig + eps
        loss_pos = scalar_loss(x_arr)
        layer.weights.data[idx] = orig - eps
        loss_neg = scalar_loss(x_arr)
        layer.weights.data[idx] = orig
        grad_w_num[idx] = (loss_pos - loss_neg) / (2.0 * eps)

    # Gradient số học cho b
    grad_b_num = np.zeros_like(layer.bias.data)
    for idx in np.ndindex(layer.bias.shape):
        orig = layer.bias.data[idx]
        layer.bias.data[idx] = orig + eps
        loss_pos = scalar_loss(x_arr)
        layer.bias.data[idx] = orig - eps
        loss_neg = scalar_loss(x_arr)
        layer.bias.data[idx] = orig
        grad_b_num[idx] = (loss_pos - loss_neg) / (2.0 * eps)

    # Gradient số học cho x
    grad_x_num = np.zeros_like(x_arr)
    for idx in np.ndindex(x_arr.shape):
        orig = x_arr[idx]
        x_arr[idx] = orig + eps
        loss_pos = scalar_loss(x_arr)
        x_arr[idx] = orig - eps
        loss_neg = scalar_loss(x_arr)
        x_arr[idx] = orig
        grad_x_num[idx] = (loss_pos - loss_neg) / (2.0 * eps)

    assert _relative_error(grad_w_ana, grad_w_num) < 1e-5
    assert _relative_error(grad_b_ana, grad_b_num) < 1e-5
    assert _relative_error(grad_x_ana, grad_x_num) < 1e-5


def test_dense_r2_accumulation_cache_isolation_and_config() -> None:
    """Kiểm tra R2, cộng dồn gradient, độc lập cache và cấu hình không hợp lệ."""
    layer = Dense(
        in_features=2, out_features=2, weight_initializer=HeInitializer(seed=1)
    )
    assert layer.in_features == 2
    assert layer.out_features == 2
    assert len(layer.parameters()) == 2

    # R2: backward trước forward ném StateError
    with pytest.raises(StateError):
        layer.backward(Tensor([[1.0, 1.0]]))

    # Sửa mảng x tại chỗ sau forward không làm hỏng cache trong Dense
    raw_x = np.array([[1.0, 2.0]], dtype=np.float32)
    x = Tensor(raw_x)
    _ = layer.forward(x)
    raw_x[:] = 999.0

    grad_out = Tensor([[1.0, -1.0]], dtype=np.float32)
    _ = layer.backward(grad_out)
    expected_dw = np.array([[1.0, -1.0], [2.0, -2.0]], dtype=np.float32)
    assert np.allclose(layer.weights.grad, expected_dw)

    # Cộng dồn gradient khi gọi backward lần 2 chưa zero_grad
    _ = layer.backward(grad_out)
    assert np.allclose(layer.weights.grad, 2.0 * expected_dw)
    assert np.allclose(layer.bias.grad, [2.0, -2.0])

    # Tham số khởi tạo sai ném InvalidConfigError
    with pytest.raises(InvalidConfigError):
        Dense(in_features=0, out_features=3)
    with pytest.raises(InvalidConfigError):
        Dense(in_features=3, out_features=-2)
    with pytest.raises(InvalidConfigError):
        Dense(in_features=True, out_features=3)


def test_dense_dtype_validation_and_casting() -> None:
    """Dense chặn dtype số nguyên và tự ép đầu vào về dtype của lớp."""
    with pytest.raises(InvalidConfigError):
        Dense(in_features=2, out_features=2, dtype=np.int32)

    layer = Dense(
        in_features=2,
        out_features=3,
        weight_initializer=HeInitializer(seed=0),
        dtype=np.float32,
    )
    x_f64 = Tensor(np.ones((4, 2), dtype=np.float64))
    y = layer.forward(x_f64)
    assert y.dtype == np.float32

    grad_out_f64 = Tensor(np.ones((4, 3), dtype=np.float64))
    grad_in = layer.backward(grad_out_f64)
    assert grad_in.dtype == np.float32
    assert layer.weights.grad.dtype == np.float32
    assert layer.bias.grad.dtype == np.float32
