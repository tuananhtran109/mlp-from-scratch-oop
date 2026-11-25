"""Interface cơ sở ILossFunction cho các hàm mất mát."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypeVar

from nnlib.exceptions import ShapeMismatchError, StateError
from nnlib.tensor import Tensor

T = TypeVar("T")


class ILossFunction(ABC):
    """Interface (Vai trò: Strategy) cho các hàm mất mát trong nnlib."""

    @abstractmethod
    def forward(self, y_pred: Tensor, y_true: Tensor) -> float:
        """Tính giá trị mất mát vô hướng trung bình theo kích thước batch N."""

    @abstractmethod
    def backward(self) -> Tensor:
        """Trả về gradient dL/dy_pred (đã chia cho N) từ cache của forward."""

    def _require_cache(self, cache: T | None) -> T:
        """Kiểm tra cache đã có trước khi chạy backward và trả về cache."""
        # Rule R2
        if cache is None:
            raise StateError(
                f"{self.__class__.__name__}: backward() phải được gọi sau " "forward()."
            )
        return cache

    @staticmethod
    def _validate_inputs(
        y_pred: Tensor,
        y_true: Tensor,
        *,
        require_2d: bool = False,
    ) -> None:
        """Kiểm tra khớp shape, số chiều và kích thước batch hợp lệ."""
        if require_2d:
            if y_pred.ndim != 2:
                raise ShapeMismatchError(
                    f"Kỳ vọng y_pred 2 chiều (N, C), nhận được {y_pred.shape}."
                )
            if y_pred.shape[0] <= 0 or y_pred.shape[1] <= 0:
                raise ShapeMismatchError(
                    f"Kích thước batch và số lớp phải > 0, nhận {y_pred.shape}."
                )
        elif y_pred.ndim == 0 or y_pred.shape[0] <= 0:
            raise ShapeMismatchError(
                f"Kỳ vọng y_pred tối thiểu 1 chiều với N > 0, nhận {y_pred.shape}."
            )
        y_true.check_shape(y_pred.shape)
