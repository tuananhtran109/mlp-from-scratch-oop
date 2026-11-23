"""Lớp chính quy hóa Dropout (inverted dropout)."""

from __future__ import annotations

import math
import numbers

import numpy as np

from nnlib.exceptions import InvalidConfigError
from nnlib.layers.base import Layer
from nnlib.tensor import Tensor


class Dropout(Layer):
    """Lớp Dropout tắt ngẫu nhiên nơ-ron khi train (Vai trò: Leaf trong Composite).

    Lưu ý khi ghép nhiều lớp Dropout trong cùng một mạng (như ``ModelBuilder``):
        Hai lớp có cùng ``seed`` và cùng ``shape`` đầu vào sẽ sinh mask giống
        hệt nhau. Cần truyền ``seed + chỉ_số_layer`` cho từng lớp để tránh
        tương quan mask.
    """

    def __init__(
        self,
        rate: numbers.Real = 0.5,
        seed: numbers.Integral | None = None,
        name: str | None = None,
    ) -> None:
        """Khởi tạo Dropout với tỷ lệ tắt 0 <= rate < 1 và seed tùy chọn."""
        super().__init__(name=name)
        # Rule R3
        if isinstance(rate, (bool, np.bool_)) or not isinstance(rate, numbers.Real):
            raise InvalidConfigError(
                "Tham số rate phải là số thực trong [0, 1), " f"nhận được {rate!r}."
            )
        rate_float = float(rate)
        if not math.isfinite(rate_float) or not (0.0 <= rate_float < 1.0):
            raise InvalidConfigError(
                f"Tham số rate phải thỏa 0 <= rate < 1, nhận được {rate!r}."
            )

        seed_int: int | None = None
        if seed is not None:
            if isinstance(seed, (bool, np.bool_)) or not isinstance(
                seed, numbers.Integral
            ):
                raise InvalidConfigError("Tham số seed phải là số nguyên hoặc None.")
            seed_int = int(seed)

        self._rate: float = rate_float
        self._seed: int | None = seed_int
        self._rng: np.random.Generator = np.random.default_rng(seed_int)
        self._mask: np.ndarray | None = None
        self._cache_meta: tuple[tuple[int, ...], np.dtype] | None = None

    @property
    def rate(self) -> float:
        """Trả về xác suất tắt nơ-ron p trong [0, 1)."""
        return self._rate

    @property
    def seed(self) -> int | None:
        """Trả về giá trị seed đã cấu hình cho bộ sinh số ngẫu nhiên."""
        return self._seed

    def forward(self, x: Tensor) -> Tensor:
        """Áp dụng inverted dropout khi train; trả nguyên đầu vào khi eval."""
        x_arr = x.data
        self._cache_meta = (x_arr.shape, x_arr.dtype)

        # Rule R3
        if not self._training or self._rate == 0.0:
            self._mask = None
            return Tensor(x_arr.copy())

        keep_prob = 1.0 - self._rate
        scale = x_arr.dtype.type(1.0 / keep_prob)
        kept = self._rng.random(size=x_arr.shape) >= self._rate
        self._mask = kept.astype(x_arr.dtype) * scale
        return Tensor(x_arr * self._mask)

    def backward(self, grad_out: Tensor) -> Tensor:
        """Nhân gradient ngược với mask đã lưu ở lần forward gần nhất."""
        # Rule R2
        expected_shape, expected_dtype = self._require_cache(self._cache_meta)
        grad_out.check_shape(expected_shape)
        g = grad_out.data.astype(expected_dtype, copy=False)

        if self._mask is None:
            return Tensor(g.copy())
        return Tensor(g * self._mask)
