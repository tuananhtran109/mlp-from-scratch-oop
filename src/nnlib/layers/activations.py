"""Lớp trừu tượng Activation và các hàm kích hoạt ReLU, Sigmoid, Tanh, Softmax."""

from __future__ import annotations

import numpy as np

from nnlib.exceptions import InvalidConfigError, ShapeMismatchError
from nnlib.layers.base import Layer
from nnlib.tensor import Tensor


class Activation(Layer):
    """Lớp cha trừu tượng cho các hàm kích hoạt (không có tham số học được)."""

    def __init__(self, name: str | None = None) -> None:
        """Khởi tạo lớp kích hoạt với cache rỗng."""
        super().__init__(name=name)
        self._cache: np.ndarray | None = None

    @staticmethod
    def _prepare_grad(
        grad_out: Tensor,
        ref: np.ndarray,
        dtype: np.dtype | None = None,
    ) -> np.ndarray:
        """Kiểm tra shape grad_out khớp ref và ép về kiểu dữ liệu mục tiêu."""
        grad_out.check_shape(ref.shape)
        target_dtype = ref.dtype if dtype is None else dtype
        return grad_out.data.astype(target_dtype, copy=False)


class ReLU(Activation):
    """Hàm kích hoạt ReLU (Vai trò: Leaf trong mẫu Composite)."""

    def __init__(self, name: str | None = None) -> None:
        """Khởi tạo lớp ReLU."""
        super().__init__(name=name)
        self._input_dtype: np.dtype | None = None

    def forward(self, x: Tensor) -> Tensor:
        """Tính y = max(0, x) và lưu mask (x > 0) cho lan truyền ngược."""
        x_arr = x.data
        self._cache = x_arr > 0
        self._input_dtype = x_arr.dtype
        out = np.maximum(x_arr, x_arr.dtype.type(0))
        return Tensor(out)

    def backward(self, grad_out: Tensor) -> Tensor:
        """Tính dL/dx = grad_out * 1[x > 0] (bằng 0 tại x <= 0)."""
        # Rule R2
        mask = self._require_cache(self._cache)
        target_dtype = self._require_cache(self._input_dtype)
        g = self._prepare_grad(grad_out, mask, dtype=target_dtype)
        grad_in = np.where(mask, g, target_dtype.type(0))
        return Tensor(grad_in)


class Sigmoid(Activation):
    """Hàm kích hoạt Sigmoid (Vai trò: Leaf trong mẫu Composite)."""

    def forward(self, x: Tensor) -> Tensor:
        """Tính y = 1 / (1 + exp(-x)) ổn định số và lưu bản sao y vào cache."""
        x_arr = x.data
        pos_mask = x_arr >= 0
        y_arr = np.empty_like(x_arr)
        one = x_arr.dtype.type(1)

        y_arr[pos_mask] = one / (one + np.exp(-x_arr[pos_mask]))
        exp_neg = np.exp(x_arr[~pos_mask])
        y_arr[~pos_mask] = exp_neg / (one + exp_neg)

        self._cache = y_arr.copy()
        return Tensor(y_arr)

    def backward(self, grad_out: Tensor) -> Tensor:
        """Tính dL/dx = grad_out * y * (1 - y)."""
        # Rule R2
        y = self._require_cache(self._cache)
        g = self._prepare_grad(grad_out, y)
        return Tensor(g * y * (y.dtype.type(1) - y))


class Tanh(Activation):
    """Hàm kích hoạt Tanh (Vai trò: Leaf trong mẫu Composite)."""

    def forward(self, x: Tensor) -> Tensor:
        """Tính y = tanh(x) và lưu bản sao y vào cache."""
        y_arr = np.tanh(x.data)
        self._cache = y_arr.copy()
        return Tensor(y_arr)

    def backward(self, grad_out: Tensor) -> Tensor:
        """Tính dL/dx = grad_out * (1 - y^2)."""
        # Rule R2
        y = self._require_cache(self._cache)
        g = self._prepare_grad(grad_out, y)
        return Tensor(g * (y.dtype.type(1) - y * y))


class Softmax(Activation):
    """Hàm kích hoạt Softmax ổn định số (Vai trò: Leaf trong mẫu Composite)."""

    def __init__(self, axis: int = -1, name: str | None = None) -> None:
        """Khởi tạo Softmax với trục chuẩn hóa xác suất (mặc định -1)."""
        super().__init__(name=name)
        if isinstance(axis, bool) or not isinstance(axis, int):
            raise InvalidConfigError(
                f"Tham số axis phải là số nguyên, nhận được {axis!r}."
            )
        self._axis: int = axis

    @property
    def axis(self) -> int:
        """Trả về trục tính phân phối xác suất Softmax."""
        return self._axis

    def forward(self, x: Tensor) -> Tensor:
        """Tính Softmax trừ giá trị lớn nhất mỗi hàng trước khi lấy exp."""
        if x.ndim == 0 or self._axis < -x.ndim or self._axis >= x.ndim:
            raise ShapeMismatchError(
                f"Trục axis={self._axis} không hợp lệ cho Tensor " f"shape {x.shape}."
            )
        # Rule R4
        x_arr = x.data
        shifted = x_arr - np.max(x_arr, axis=self._axis, keepdims=True)
        exp_vals = np.exp(shifted)
        y_arr = exp_vals / np.sum(exp_vals, axis=self._axis, keepdims=True)
        self._cache = y_arr.copy()
        return Tensor(y_arr)

    def backward(self, grad_out: Tensor) -> Tensor:
        """Tính dL/dz = p * (g - sum_j(g_j * p_j)) theo trục axis."""
        # Rule R2
        p = self._require_cache(self._cache)
        g = self._prepare_grad(grad_out, p)
        dot = np.sum(g * p, axis=self._axis, keepdims=True)
        return Tensor(p * (g - dot))
