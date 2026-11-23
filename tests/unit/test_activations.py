"""Kiểm thử đơn vị cho các hàm kích hoạt (bao gồm T04, T05, T06)."""

import numpy as np
import pytest

from nnlib.exceptions import InvalidConfigError, ShapeMismatchError, StateError
from nnlib.layers import Activation, ReLU, Sigmoid, Softmax, Tanh
from nnlib.tensor import Tensor


def _relative_error(analytical: np.ndarray, numerical: np.ndarray) -> float:
    """Tính sai số tương đối cực đại theo đặc tả gradient check."""
    diff = np.abs(analytical - numerical)
    denom = np.maximum(np.abs(analytical) + np.abs(numerical), 1e-12)
    return float(np.max(diff / denom))


def _numerical_grad_input(
    layer: Activation,
    x_arr: np.ndarray,
    u_arr: np.ndarray,
    eps: float = 1e-5,
) -> np.ndarray:
    """Tính gradient số học theo đầu vào x bằng sai phân trung tâm."""
    grad_num = np.zeros_like(x_arr, dtype=np.float64)
    x_work = x_arr.astype(np.float64, copy=True)

    for idx in np.ndindex(x_work.shape):
        orig = x_work[idx]
        x_work[idx] = orig + eps
        loss_pos = float(
            np.sum(layer.forward(Tensor(x_work, dtype=np.float64)).data * u_arr)
        )
        x_work[idx] = orig - eps
        loss_neg = float(
            np.sum(layer.forward(Tensor(x_work, dtype=np.float64)).data * u_arr)
        )
        x_work[idx] = orig
        grad_num[idx] = (loss_pos - loss_neg) / (2.0 * eps)

    return grad_num


def test_t04_relu_forward_and_backward_values() -> None:
    """T04: ReLU với số âm, 0, dương cho đúng output và gradient = 0 khi x <= 0."""
    relu = ReLU()
    assert relu.parameters() == []

    x = Tensor([[-2.5, 0.0, 3.0], [-0.1, 1.5, 0.0]], dtype=np.float32)
    y = relu.forward(x)
    assert y.dtype == np.float32
    assert np.allclose(y.data, [[0.0, 0.0, 3.0], [0.0, 1.5, 0.0]])

    grad_out = Tensor([[1.0, 2.0, 3.0], [4.0, -5.0, 6.0]], dtype=np.float32)
    grad_in = relu.backward(grad_out)
    assert grad_in.dtype == np.float32
    assert np.allclose(grad_in.data, [[0.0, 0.0, 3.0], [0.0, -5.0, 0.0]])


@pytest.mark.parametrize(
    "layer_cls",
    [Sigmoid, Tanh, ReLU, Softmax],
)
def test_t05_activations_gradient_check_float64(
    layer_cls: type[Activation],
) -> None:
    """T05 & M2: Gradient check Sigmoid, Tanh, ReLU, Softmax có sai số < 1e-5."""
    rng = np.random.default_rng(2026)
    layer = layer_cls()

    # Dịch nhẹ khỏi 0 để điểm không khả vi của ReLU không rơi vào [x-eps, x+eps]
    x_arr = rng.normal(0.0, 1.0, size=(4, 5)).astype(np.float64)
    x_arr = np.where(np.abs(x_arr) < 0.05, 0.25, x_arr)
    u_arr = rng.normal(0.0, 1.0, size=(4, 5)).astype(np.float64)

    _ = layer.forward(Tensor(x_arr, dtype=np.float64))
    grad_ana = layer.backward(Tensor(u_arr, dtype=np.float64)).data
    grad_num = _numerical_grad_input(layer, x_arr, u_arr, eps=1e-5)

    assert _relative_error(grad_ana, grad_num) < 1e-5


def test_t06_softmax_numerical_stability_large_inputs() -> None:
    """T06 (R4): Softmax với đầu vào lớn (~1000) tổng hàng = 1, không ra NaN."""
    softmax = Softmax(axis=-1)
    assert softmax.axis == -1

    x = Tensor(
        [[1000.0, 1001.0, 999.0], [-1000.0, -999.0, -1001.0]],
        dtype=np.float32,
    )
    y = softmax.forward(x)

    assert y.dtype == np.float32
    assert not np.any(np.isnan(y.data))
    assert not np.any(np.isinf(y.data))
    assert np.allclose(np.sum(y.data, axis=-1), 1.0, atol=1e-6)
    assert np.allclose(y.data[0], y.data[1], atol=1e-6)


@pytest.mark.parametrize(
    "layer_cls",
    [ReLU, Sigmoid, Tanh, Softmax],
)
def test_activations_r2_cache_isolation_and_shape_errors(
    layer_cls: type[Activation],
) -> None:
    """Kiểm tra ABC, R2 (StateError), độc lập cache, sai shape và giữ float32."""
    with pytest.raises(TypeError):
        Activation()  # type: ignore[abstract]

    layer = layer_cls()
    with pytest.raises(StateError):
        layer.backward(Tensor([[1.0, 2.0]]))

    raw_x = np.array([[0.5, -0.5], [1.0, 2.0]], dtype=np.float32)
    y = layer.forward(Tensor(raw_x))
    assert y.dtype == np.float32

    # Sửa input/output tại chỗ không làm sai lệch backward
    raw_x[:] = 999.0
    y.data[:] = -999.0

    # So sánh backward với một bản chạy sạch để chứng minh cache độc lập
    clean_layer = layer_cls()
    _ = clean_layer.forward(Tensor([[0.5, -0.5], [1.0, 2.0]], dtype=np.float32))
    grad_out_f64 = Tensor(np.array([[1.0, -1.0], [0.5, 0.25]], dtype=np.float64))

    grad_a = layer.backward(grad_out_f64)
    grad_b = clean_layer.backward(grad_out_f64)
    assert grad_a.dtype == np.float32
    assert np.allclose(grad_a.data, grad_b.data)

    with pytest.raises(ShapeMismatchError):
        layer.backward(Tensor(np.ones((2, 3), dtype=np.float32)))


def test_sigmoid_extremes_and_softmax_invalid_axis() -> None:
    """Sigmoid ổn định ở đầu vào rất lớn/nhỏ; Softmax kiểm tra axis hợp lệ."""
    sig = Sigmoid()
    y = sig.forward(Tensor([[-1000.0, 0.0, 1000.0]], dtype=np.float32))
    assert not np.any(np.isnan(y.data))
    assert np.allclose(y.data, [[0.0, 0.5, 1.0]])

    with pytest.raises(InvalidConfigError):
        Softmax(axis=True)
    with pytest.raises(InvalidConfigError):
        Softmax(axis=1.5)  # type: ignore[arg-type]

    sm = Softmax(axis=2)
    with pytest.raises(ShapeMismatchError):
        sm.forward(Tensor([[1.0, 2.0]], dtype=np.float32))
