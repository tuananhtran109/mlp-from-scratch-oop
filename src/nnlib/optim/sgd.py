"""Thuật toán tối ưu hóa Gradient Descent ngẫu nhiên (SGD) có hỗ trợ Momentum."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from nnlib.exceptions import InvalidConfigError
from nnlib.optim.base import IOptimizer
from nnlib.tensor import Parameter


class SGD(IOptimizer):
    """Bộ tối ưu hóa SGD (Vai trò: Concrete Strategy trong mẫu Strategy)."""

    def __init__(self, lr: float = 0.1, momentum: float = 0.9) -> None:
        """Khởi tạo SGD với tốc độ học lr và hệ số momentum (mặc định lr=0.1, mu=0.9).

        Quy tắc R7:
            Hệ số momentum phải thuộc khoảng [0.0, 1.0) và không nhận kiểu boolean.
        """
        super().__init__(lr=lr)
        # Rule R7
        if (
            isinstance(momentum, bool)
            or not isinstance(momentum, (int, float))
            or np.isnan(momentum)
            or not (0.0 <= float(momentum) < 1.0)
        ):
            raise InvalidConfigError(
                f"Hệ số momentum phải thuộc khoảng [0, 1), nhận được: {momentum!r}."
            )
        self._momentum: float = float(momentum)

        # Lưu vận tốc theo chỉ số (index int) của tham số trong danh sách params
        self._velocities: dict[int, np.ndarray] = {}

    @property
    def momentum(self) -> float:
        """Trả về hệ số momentum."""
        return self._momentum

    def step(self, params: Sequence[Parameter]) -> None:
        """Cập nhật trọng số theo công thức SGD có/không có momentum.

        Công thức:
            v <- mu * v - eta * grad
            theta <- theta + v
        """
        # Rule R7
        for idx, param in enumerate(params):
            if idx not in self._velocities:
                self._velocities[idx] = np.zeros_like(param.data)

            v_old = self._velocities[idx]
            v_new = self._momentum * v_old - self._lr * param.grad
            self._velocities[idx] = v_new

            # Cập nhật mảng tham số thông qua setter hoặc in-place
            param.data = param.data + v_new
