"""Lớp FakeLayer phục vụ kiểm thử độc lập cho Optimizer và Pipeline."""

from __future__ import annotations

import numpy as np

from nnlib.tensor import Parameter, Tensor


class FakeLayer:
    """Lớp mạng giả lập sử dụng Parameter thật của Người A."""

    def __init__(self, in_features: int = 2, out_features: int = 2) -> None:
        """Khởi tạo FakeLayer với ma trận trọng số và bias thật."""
        self._weights = Parameter(
            np.ones((in_features, out_features), dtype=np.float32),
            name="fake_weights",
        )
        self._bias = Parameter(
            np.zeros(out_features, dtype=np.float32),
            name="fake_bias",
        )

    @property
    def weights(self) -> Parameter:
        """Trả về tham số trọng số."""
        return self._weights

    @property
    def bias(self) -> Parameter:
        """Trả về tham số bias."""
        return self._bias

    def parameters(self) -> list[Parameter]:
        """Trả về danh sách các Parameter thật để optimizer cập nhật."""
        return [self._weights, self._bias]

    def forward(self, x: Tensor) -> Tensor:
        """Lan truyền thuận đơn giản y = x @ W + b."""
        return x @ self._weights + self._bias
