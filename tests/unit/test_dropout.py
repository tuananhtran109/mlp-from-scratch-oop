"""Kiểm thử đơn vị cho lớp Dropout (bao gồm T07 và quy tắc R2, R3)."""

import numpy as np
import pytest

from nnlib.exceptions import InvalidConfigError, ShapeMismatchError, StateError
from nnlib.layers import Dropout
from nnlib.tensor import Tensor


def test_t07_dropout_eval_and_train_statistics() -> None:
    """T07 (R3): Eval trả nguyên; train tắt ~p và giữ nguyên trung bình."""
    rate = 0.2
    dropout = Dropout(rate=rate, seed=42)
    assert dropout.rate == rate
    assert dropout.seed == 42
    assert dropout.parameters() == []

    x_arr = np.full((1000, 100), 2.5, dtype=np.float32)
    x = Tensor(x_arr)

    # Chế độ train: tỷ lệ phần tử bằng 0 xấp xỉ rate, trung bình giữ nguyên
    dropout.train()
    y_train = dropout.forward(x)
    assert y_train.dtype == np.float32
    zero_ratio = float(np.mean(y_train.data == 0.0))
    assert abs(zero_ratio - rate) < 0.01
    assert abs(float(np.mean(y_train.data)) - 2.5) < 0.05

    # Các phần tử còn sống được nhân đúng 1 / (1 - rate)
    nonzero_vals = y_train.data[y_train.data != 0.0]
    assert np.allclose(nonzero_vals, 2.5 / (1.0 - rate))

    # Chế độ eval: trả nguyên đầu vào (và không dùng chung bộ nhớ với x)
    dropout.eval()
    y_eval = dropout.forward(x)
    assert y_eval.dtype == np.float32
    assert np.array_equal(y_eval.data, x_arr)
    y_eval.data[0, 0] = -1.0
    assert x.data[0, 0] == 2.5


def test_dropout_backward_reproducibility_and_mode_switch() -> None:
    """Kiểm tra backward khớp mask, tái lập theo seed và chuyển train/eval."""
    d1 = Dropout(rate=0.4, seed=123)
    d2 = Dropout(rate=0.4, seed=123)

    ones = Tensor(np.ones((8, 16), dtype=np.float32))
    y1 = d1.forward(ones)
    y2 = d2.forward(ones)
    assert np.array_equal(y1.data, y2.data)

    # Với đầu vào toàn 1, y1.data chính là mask đã nhân scale
    grad_out_f64 = Tensor(np.arange(8 * 16, dtype=np.float64).reshape(8, 16) * 0.1)
    grad_in = d1.backward(grad_out_f64)
    assert grad_in.dtype == np.float32
    expected = grad_out_f64.data.astype(np.float32) * y1.data
    assert np.allclose(grad_in.data, expected)

    # Chuyển sang eval và forward lại: backward không được dùng nhầm mask cũ
    d1.eval()
    _ = d1.forward(ones)
    grad_eval = d1.backward(grad_out_f64)
    assert grad_eval.dtype == np.float32
    assert np.allclose(grad_eval.data, grad_out_f64.data.astype(np.float32))

    # rate = 0.0 ở train cũng truyền nguyên đầu vào và gradient
    d_zero = Dropout(rate=0.0, seed=0)
    y_zero = d_zero.forward(Tensor([[1.0, -2.0]], dtype=np.float32))
    assert np.allclose(y_zero.data, [[1.0, -2.0]])
    g_zero = d_zero.backward(Tensor([[3.0, 4.0]], dtype=np.float32))
    assert np.allclose(g_zero.data, [[3.0, 4.0]])


def test_dropout_r2_shape_and_invalid_config() -> None:
    """Kiểm tra R2 (StateError), sai shape grad_out và cấu hình rate/seed sai."""
    dropout = Dropout(rate=0.3, seed=7)
    with pytest.raises(StateError):
        dropout.backward(Tensor([[1.0, 2.0]]))

    _ = dropout.forward(Tensor([[1.0, 2.0]], dtype=np.float32))
    with pytest.raises(ShapeMismatchError):
        dropout.backward(Tensor([[1.0, 2.0, 3.0]], dtype=np.float32))

    for bad_rate in (-0.1, 1.0, 1.5, float("nan"), float("inf"), True, "0.2"):
        with pytest.raises(InvalidConfigError):
            Dropout(rate=bad_rate)  # type: ignore[arg-type]

    with pytest.raises(InvalidConfigError):
        Dropout(rate=0.2, seed=True)
    with pytest.raises(InvalidConfigError):
        Dropout(rate=0.2, seed="42")  # type: ignore[arg-type]


def test_dropout_accepts_numpy_scalars_and_rejects_np_bool() -> None:
    """Dropout chấp nhận số NumPy cho rate/seed nhưng vẫn chặn np.bool_."""
    d_np = Dropout(rate=np.float32(0.2), seed=np.int64(1))
    d_py = Dropout(rate=0.2, seed=1)
    assert isinstance(d_np.rate, float)
    assert abs(d_np.rate - 0.2) < 1e-6
    assert d_np.seed == 1

    x = Tensor(np.ones((4, 8), dtype=np.float32))
    assert np.array_equal(d_np.forward(x).data, d_py.forward(x).data)

    with pytest.raises(InvalidConfigError):
        Dropout(rate=np.bool_(True))  # type: ignore[arg-type]
    with pytest.raises(InvalidConfigError):
        Dropout(rate=0.2, seed=np.bool_(False))  # type: ignore[arg-type]
