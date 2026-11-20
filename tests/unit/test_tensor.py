"""Kiểm thử đơn vị cho Tensor, Parameter và hệ ngoại lệ."""

import numpy as np
import pytest

from nnlib.exceptions import (
    InvalidConfigError,
    ModelFileError,
    NNError,
    ShapeMismatchError,
    StateError,
)
from nnlib.tensor import Parameter, Tensor


def test_exceptions_hierarchy() -> None:
    """Mọi ngoại lệ con đều kế thừa NNError và giữ thông điệp."""
    for exc_cls in (
        ShapeMismatchError,
        InvalidConfigError,
        StateError,
        ModelFileError,
    ):
        err = exc_cls("lỗi mẫu")
        assert isinstance(err, NNError)
        assert err.message == "lỗi mẫu"


def test_tensor_creation_and_dtypes() -> None:
    """List Python mặc định float32; ndarray/Tensor float64 được giữ nguyên."""
    t_int_list = Tensor([[1, 2], [3, 4]])
    t_float_list = Tensor([[1.0, 2.0], [3.0, 4.0]])
    assert t_int_list.shape == (2, 2)
    assert t_int_list.dtype == np.float32
    assert t_float_list.dtype == np.float32

    t_f64 = Tensor(np.ones((2, 3), dtype=np.float64))
    assert t_f64.dtype == np.float64
    assert Tensor(t_f64).dtype == np.float64
    assert Tensor([[1.0, 2.0]], dtype=np.float64).dtype == np.float64

    with pytest.raises(InvalidConfigError):
        Tensor(["a", "b"])
    with pytest.raises(InvalidConfigError):
        Tensor([True, False])


def test_tensor_copy_transpose_and_data_setter() -> None:
    """Kiểm tra copy độc lập, chuyển vị T, bọc Tensor(Tensor) và data setter."""
    raw = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    t_view = Tensor(raw)
    t_copied = Tensor(raw, copy=True)
    t_from_tensor = Tensor(t_copied, copy=True)
    t_clone = t_copied.copy()

    raw[0, 0] = 99.0
    assert t_view.data[0, 0] == 99.0
    assert t_copied.data[0, 0] == 1.0

    t_copied.data[0, 0] = -5.0
    assert t_from_tensor.data[0, 0] == 1.0
    assert t_clone.data[0, 0] == 1.0

    assert t_clone.T.shape == (3, 2)
    assert np.allclose(t_clone.T.data, [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])

    with pytest.raises(ShapeMismatchError):
        t_clone.data = np.zeros((3, 2), dtype=np.float32)


def test_tensor_operations_and_shape_check() -> None:
    """Kiểm tra phép toán hai chiều (trái/phải), đổi dấu và báo lỗi sai shape."""
    a = Tensor([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    b = Tensor([[2.0, 0.0], [1.0, 2.0]], dtype=np.float32)
    arr = np.array([[2.0, 2.0], [2.0, 2.0]], dtype=np.float32)

    assert np.allclose((a + b).data, [[3.0, 2.0], [4.0, 6.0]])
    assert np.allclose((a - b).data, [[-1.0, 2.0], [2.0, 2.0]])
    assert np.allclose((a * b).data, [[2.0, 0.0], [3.0, 8.0]])
    assert np.allclose((a @ b).data, [[4.0, 4.0], [10.0, 8.0]])
    assert np.allclose((-a).data, [[-1.0, -2.0], [-3.0, -4.0]])

    # Toán tử bên phải với hằng số và ndarray
    assert np.allclose((2.0 + a).data, [[3.0, 4.0], [5.0, 6.0]])
    assert np.allclose((10.0 - a).data, [[9.0, 8.0], [7.0, 6.0]])
    assert np.allclose((2.0 * a).data, [[2.0, 4.0], [6.0, 8.0]])
    assert np.allclose((arr * a).data, [[2.0, 4.0], [6.0, 8.0]])
    assert np.allclose((arr @ a).data, [[8.0, 12.0], [8.0, 12.0]])
    assert (2.0 * a).dtype == np.float32

    a.check_shape((None, 2))
    with pytest.raises(ShapeMismatchError):
        a.check_shape((None, 3))
    with pytest.raises(ShapeMismatchError):
        _ = a @ Tensor(np.zeros((3, 2), dtype=np.float32))


def test_parameter_grad_accumulation_zero_grad_and_dtype() -> None:
    """Parameter chỉ nhận số thực, hỗ trợ cộng dồn gradient và zero_grad."""
    w = Parameter(np.ones((2, 3), dtype=np.float32), name="weights")
    assert isinstance(w, Tensor)
    assert w.name == "weights"
    assert w.grad.shape == (2, 3)
    assert np.all(w.grad == 0.0)

    w.grad += np.full((2, 3), 0.5, dtype=np.float32)
    w.grad += np.full((2, 3), 0.25, dtype=np.float32)
    assert np.allclose(w.grad, 0.75)

    w.data -= 0.1 * w.grad
    assert np.allclose(w.data, 0.925)

    w.zero_grad()
    assert np.all(w.grad == 0.0)

    with pytest.raises(ShapeMismatchError):
        w.grad = np.zeros((3, 2), dtype=np.float32)

    with pytest.raises(InvalidConfigError):
        Parameter(np.ones(2), dtype=np.int32)
