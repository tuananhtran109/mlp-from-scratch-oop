"""Các phép biến đổi dữ liệu (Transforms) phục vụ tiền xử lý và mã hóa nhãn."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from nnlib.exceptions import InvalidConfigError


class ITransform(ABC):
    """Giao diện trừu tượng cho các phép biến đổi dữ liệu."""

    @abstractmethod
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Thực hiện biến đổi trên mảng NumPy và trả về kết quả."""


class OneHotEncoder(ITransform):
    """Mã hóa nhãn số nguyên thành vector one-hot."""

    def __init__(self, num_classes: int = 10) -> None:
        """Khởi tạo bộ mã hóa với số lượng lớp mục tiêu."""
        if (
            isinstance(num_classes, bool)
            or not isinstance(num_classes, (int, np.integer))
            or int(num_classes) <= 0
        ):
            raise InvalidConfigError(
                f"num_classes phải là số nguyên dương, nhận được: {num_classes!r}."
            )
        self._num_classes: int = int(num_classes)

    @property
    def num_classes(self) -> int:
        """Trả về số lượng lớp mục tiêu."""
        return self._num_classes

    def __call__(self, labels: np.ndarray) -> np.ndarray:
        """Chuyển đổi mảng 1D nhãn nguyên (N,) sang 2D one-hot (N, num_classes)."""
        arr = np.asarray(labels, dtype=np.int64)
        if arr.ndim != 1:
            raise InvalidConfigError(
                f"OneHotEncoder chỉ nhận mảng nhãn 1 chiều (N,), "
                f"nhận được: {arr.shape}."
            )

        if len(arr) == 0:
            return np.zeros((0, self._num_classes), dtype=np.float32)

        min_val = arr.min()
        max_val = arr.max()
        if min_val < 0 or max_val >= self._num_classes:
            raise InvalidConfigError(
                f"Nhãn nằm ngoài khoảng hợp lệ [0, {self._num_classes - 1}], "
                f"nhận được giá trị trong khoảng [{min_val}, {max_val}]."
            )

        one_hot = np.zeros((len(arr), self._num_classes), dtype=np.float32)
        one_hot[np.arange(len(arr)), arr] = 1.0
        return one_hot


class Normalizer(ITransform):
    """Chuẩn hóa giá trị điểm ảnh về khoảng [0.0, 1.0]."""

    def __init__(self, scale: float = 255.0) -> None:
        """Khởi tạo với hệ số chia (mặc định 255.0)."""
        if (
            isinstance(scale, bool)
            or not isinstance(scale, (int, float))
            or float(scale) <= 0.0
        ):
            raise InvalidConfigError(
                f"Hệ số scale phải là số dương, nhận được: {scale!r}."
            )
        self._scale: float = float(scale)

    @property
    def scale(self) -> float:
        """Trả về hệ số chia chuẩn hóa."""
        return self._scale

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Chia mảng điểm ảnh cho hệ số scale để đưa về float32 [0, 1]."""
        return np.asarray(x, dtype=np.float32) / self._scale
