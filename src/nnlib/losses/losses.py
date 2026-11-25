"""Các hàm mất mát cụ thể: MSELoss và CrossEntropyLoss."""

from __future__ import annotations

import math
import numbers

import numpy as np

from nnlib.exceptions import InvalidConfigError
from nnlib.losses.base import ILossFunction
from nnlib.tensor import Tensor


class MSELoss(ILossFunction):
    """Hàm mất mát sai số bình phương trung bình (Vai trò: ConcreteStrategy)."""

    def __init__(self) -> None:
        """Khởi tạo MSELoss với cache rỗng."""
        self._cache: tuple[np.ndarray, np.dtype] | None = None

    def forward(self, y_pred: Tensor, y_true: Tensor) -> float:
        """Tính L_MSE = (1 / N) * sum_n ||y_pred_n - y_true_n||^2."""
        self._validate_inputs(y_pred, y_true, require_2d=False)

        p_arr = y_pred.data.astype(np.float64, copy=True)
        y_arr = y_true.data.astype(np.float64, copy=True)
        diff = p_arr - y_arr
        batch_size = diff.shape[0]

        self._cache = (diff, y_pred.dtype)
        return float(np.sum(diff * diff) / batch_size)

    def backward(self) -> Tensor:
        """Trả về dL/dy_pred = (2 / N) * (y_pred - y_true) cùng dtype y_pred."""
        # Rule R2
        diff, out_dtype = self._require_cache(self._cache)
        batch_size = diff.shape[0]
        grad = (2.0 / batch_size) * diff
        return Tensor(grad.astype(out_dtype, copy=False))


class CrossEntropyLoss(ILossFunction):
    """Hàm mất mát Cross-Entropy ổn định số (Vai trò: ConcreteStrategy)."""

    def __init__(self, eps: numbers.Real = 1e-12) -> None:
        """Khởi tạo CrossEntropyLoss với ngưỡng kẹp eps trong (0, 0.5)."""
        if isinstance(eps, (bool, np.bool_)) or not isinstance(eps, numbers.Real):
            raise InvalidConfigError(
                f"Tham số eps phải là số thực trong (0, 0.5), nhận được {eps!r}."
            )
        eps_float = float(eps)
        if not math.isfinite(eps_float) or not (0.0 < eps_float < 0.5):
            raise InvalidConfigError(
                f"Tham số eps phải thỏa 0 < eps < 0.5, nhận được {eps!r}."
            )
        self._eps: float = eps_float
        self._cache: tuple[np.ndarray, np.ndarray, np.dtype] | None = None

    @property
    def eps(self) -> float:
        """Trả về hằng số epsilon dùng để kẹp xác suất vào [eps, 1 - eps]."""
        return self._eps

    def forward(self, y_pred: Tensor, y_true: Tensor) -> float:
        """Tính Cross-Entropy với xác suất kẹp trong [eps, 1 - eps] (R4)."""
        self._validate_inputs(y_pred, y_true, require_2d=True)

        # Rule R4
        p_arr = y_pred.data.astype(np.float64, copy=True)
        y_arr = y_true.data.astype(np.float64, copy=True)
        p_clipped = np.clip(p_arr, self._eps, 1.0 - self._eps)
        batch_size = p_arr.shape[0]

        self._cache = (p_clipped, y_arr, y_pred.dtype)
        return float(-np.sum(y_arr * np.log(p_clipped)) / batch_size)

    def backward(self) -> Tensor:
        """Trả về dL/dp = -(y_true / p_clipped) / N cùng dtype y_pred."""
        # Rule R2
        p_clipped, y_arr, out_dtype = self._require_cache(self._cache)
        batch_size = p_clipped.shape[0]
        grad = -(y_arr / p_clipped) / batch_size
        return Tensor(grad.astype(out_dtype, copy=False))
