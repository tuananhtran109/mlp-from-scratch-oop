"""Giao diện trừu tượng IOptimizer định nghĩa mẫu thiết kế Strategy cho tối ưu hóa."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np

from nnlib.exceptions import InvalidConfigError
from nnlib.tensor import Parameter


class IOptimizer(ABC):
    """Giao diện Strategy cho các thuật toán cập nhật trọng số mô hình."""

    def __init__(self, lr: float = 0.1) -> None:
        """Khởi tạo bộ tối ưu với tốc độ học (learning rate).

        Quy tắc R7:
            Tốc độ học (lr) phải là số thực dương (> 0) và không nhận kiểu boolean.
        """
        # Rule R7
        if (
            isinstance(lr, bool)
            or not isinstance(lr, (int, float))
            or np.isnan(lr)
            or np.isinf(lr)
            or float(lr) <= 0.0
        ):
            raise InvalidConfigError(
                f"Tốc độ học (lr) phải là số thực dương, nhận được: {lr!r}."
            )
        self._lr: float = float(lr)

    @property
    def lr(self) -> float:
        """Trả về tốc độ học của optimizer."""
        return self._lr

    @abstractmethod
    def step(self, params: Sequence[Parameter]) -> None:
        """Cập nhật giá trị trọng số cho danh sách tham số dựa trên gradient."""

    def zero_grad(self, params: Sequence[Parameter]) -> None:
        """Đặt toàn bộ gradient tích lũy của các tham số về 0."""
        for param in params:
            param.zero_grad()
