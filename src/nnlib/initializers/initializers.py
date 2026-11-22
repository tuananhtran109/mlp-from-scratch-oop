"""Các chiến lược khởi tạo trọng số: He, Xavier và Zeros."""

from __future__ import annotations

import numpy as np
from numpy.typing import DTypeLike

from nnlib.initializers.base import IInitializer


class HeInitializer(IInitializer):
    """Khởi tạo He (chuẩn cho lớp trước ReLU), độ lệch chuẩn sqrt(2 / in)."""

    def initialize(
        self,
        shape: tuple[int, ...],
        dtype: DTypeLike = np.float32,
    ) -> np.ndarray:
        """Sinh ma trận trọng số theo phân phối chuẩn N(0, sqrt(2 / fan_in))."""
        self._validate_shape(shape, min_ndim=2)
        fan_in = shape[0]
        std = float(np.sqrt(2.0 / fan_in))
        return self._rng.normal(loc=0.0, scale=std, size=shape).astype(dtype)


class XavierInitializer(IInitializer):
    """Khởi tạo Xavier/Glorot, độ lệch chuẩn sqrt(2 / (in + out))."""

    def initialize(
        self,
        shape: tuple[int, ...],
        dtype: DTypeLike = np.float32,
    ) -> np.ndarray:
        """Sinh ma trận trọng số theo N(0, sqrt(2 / (fan_in + fan_out)))."""
        self._validate_shape(shape, min_ndim=2)
        fan_in, fan_out = shape[0], shape[1]
        std = float(np.sqrt(2.0 / (fan_in + fan_out)))
        return self._rng.normal(loc=0.0, scale=std, size=shape).astype(dtype)


class ZerosInitializer(IInitializer):
    """Khởi tạo mảng toàn giá trị 0 (thường dùng cho vector bias)."""

    def initialize(
        self,
        shape: tuple[int, ...],
        dtype: DTypeLike = np.float32,
    ) -> np.ndarray:
        """Trả về mảng toàn 0 với shape và dtype chỉ định."""
        self._validate_shape(shape, min_ndim=1)
        return np.zeros(shape, dtype=dtype)
