"""Interface cơ sở cho các chiến lược khởi tạo trọng số."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import DTypeLike

from nnlib.exceptions import InvalidConfigError


class IInitializer(ABC):
    """Interface định nghĩa chiến lược khởi tạo mảng trọng số theo shape."""

    def __init__(self, seed: int | None = None) -> None:
        """Khởi tạo bộ sinh số ngẫu nhiên với seed tùy chọn để tái lập."""
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            raise InvalidConfigError("Tham số seed phải là số nguyên hoặc None.")
        self._seed: int | None = seed
        self._rng: np.random.Generator = np.random.default_rng(seed)

    @property
    def seed(self) -> int | None:
        """Trả về giá trị seed đã cấu hình."""
        return self._seed

    @staticmethod
    def _validate_shape(shape: tuple[int, ...], min_ndim: int = 1) -> None:
        """Kiểm tra shape hợp lệ (chứa các số nguyên dương và đủ số chiều)."""
        if len(shape) < min_ndim:
            raise InvalidConfigError(
                f"Khởi tạo yêu cầu tối thiểu {min_ndim} chiều, " f"nhận được {shape}."
            )
        has_invalid_dim = any(
            isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0
            for dim in shape
        )
        if has_invalid_dim:
            raise InvalidConfigError(
                "Mọi chiều trong shape phải là số nguyên dương, " f"nhận được {shape}."
            )

    @abstractmethod
    def initialize(
        self,
        shape: tuple[int, ...],
        dtype: DTypeLike = np.float32,
    ) -> np.ndarray:
        """Sinh mảng NumPy đã khởi tạo theo shape và kiểu số thực yêu cầu."""
