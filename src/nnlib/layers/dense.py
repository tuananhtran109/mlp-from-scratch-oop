"""Lớp kết nối đầy đủ Dense (y = xW + b)."""

from __future__ import annotations

import numpy as np
from numpy.typing import DTypeLike

from nnlib.exceptions import InvalidConfigError
from nnlib.initializers import HeInitializer, IInitializer, ZerosInitializer
from nnlib.layers.base import Layer
from nnlib.tensor import Parameter, Tensor


class Dense(Layer):
    """Lớp kết nối đầy đủ (Vai trò: Leaf trong mẫu Composite)."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        weight_initializer: IInitializer | None = None,
        bias_initializer: IInitializer | None = None,
        dtype: DTypeLike = np.float32,
        name: str | None = None,
    ) -> None:
        """Khởi tạo lớp Dense với kích thước vào/ra và chiến lược khởi tạo."""
        super().__init__(name=name)
        self._validate_positive_int(in_features, "in_features")
        self._validate_positive_int(out_features, "out_features")

        try:
            resolved_dtype = np.dtype(dtype)
        except TypeError as exc:
            raise InvalidConfigError(f"dtype không hợp lệ: {dtype!r}.") from exc
        if not np.issubdtype(resolved_dtype, np.floating):
            raise InvalidConfigError("dtype của Dense phải là kiểu số thực.")

        w_init = (
            weight_initializer if weight_initializer is not None else HeInitializer()
        )
        b_init = (
            bias_initializer if bias_initializer is not None else ZerosInitializer()
        )
        if not isinstance(w_init, IInitializer):
            raise InvalidConfigError("weight_initializer phải thực thi IInitializer.")
        if not isinstance(b_init, IInitializer):
            raise InvalidConfigError("bias_initializer phải thực thi IInitializer.")

        self._in_features: int = in_features
        self._out_features: int = out_features
        self._weights: Parameter = Parameter(
            w_init.initialize((in_features, out_features), dtype=resolved_dtype),
            name="weights",
            dtype=resolved_dtype,
        )
        self._bias: Parameter = Parameter(
            b_init.initialize((out_features,), dtype=resolved_dtype),
            name="bias",
            dtype=resolved_dtype,
        )
        self._cache_x: Tensor | None = None

    @staticmethod
    def _validate_positive_int(value: int, field_name: str) -> None:
        """Kiểm tra tham số kích thước phải là số nguyên dương."""
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise InvalidConfigError(
                f"{field_name} phải là số nguyên dương, nhận được {value!r}."
            )

    @property
    def in_features(self) -> int:
        """Trả về số đặc trưng đầu vào của lớp."""
        return self._in_features

    @property
    def out_features(self) -> int:
        """Trả về số đặc trưng đầu ra của lớp."""
        return self._out_features

    @property
    def weights(self) -> Parameter:
        """Trả về tham số ma trận trọng số W có shape (in_features, out_features)."""
        return self._weights

    @property
    def bias(self) -> Parameter:
        """Trả về tham số vector độ lệch b có shape (out_features,)."""
        return self._bias

    def parameters(self) -> list[Parameter]:
        """Trả về danh sách tham số học được [weights, bias]."""
        return [self._weights, self._bias]

    def forward(self, x: Tensor) -> Tensor:
        """Tính y = xW + b; tự động ép x về dtype của lớp và lưu bản sao cache."""
        # Rule R1
        x.check_shape((None, self._in_features))
        x_arr = x.data.astype(self._weights.dtype, copy=True)
        self._cache_x = Tensor(x_arr)
        out_data = x_arr @ self._weights.data + self._bias.data
        return Tensor(out_data)

    def backward(self, grad_out: Tensor) -> Tensor:
        """Cộng dồn gradient cho W, b và trả về dL/dx cùng dtype với lớp."""
        # Rule R2
        cache_x = self._require_cache(self._cache_x)

        batch_size = cache_x.shape[0]
        grad_out.check_shape((batch_size, self._out_features))

        g = grad_out.data.astype(self._weights.dtype, copy=False)
        self._weights.grad += cache_x.data.T @ g
        self._bias.grad += np.sum(g, axis=0)

        grad_in = g @ self._weights.data.T
        return Tensor(grad_in)
