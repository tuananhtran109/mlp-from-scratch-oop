"""Hàm mất mát giả lập phục vụ kiểm thử Trainer khi nnlib.losses chưa sẵn sàng."""

from __future__ import annotations

import numpy as np

from nnlib.exceptions import ShapeMismatchError, StateError
from nnlib.tensor import Tensor


class FakeMSELoss:
    """Hàm mất mát MSE giả lập khớp hợp đồng ILossFunction (dùng tạm trong test)."""

    def __init__(self) -> None:
        """Khởi tạo FakeMSELoss với cache rỗng."""
        self._cache: tuple[np.ndarray, np.dtype] | None = None

    def forward(self, y_pred: Tensor, y_true: Tensor) -> float:
        """Tính L_MSE = (1 / N) * sum_n ||y_pred_n - y_true_n||^2."""
        if y_pred.ndim == 0 or y_pred.shape[0] <= 0:
            raise ShapeMismatchError(
                f"Kỳ vọng y_pred tối thiểu 1 chiều với N > 0, nhận: {y_pred.shape}."
            )
        y_true.check_shape(y_pred.shape)

        p_arr = y_pred.data.astype(np.float64, copy=True)
        y_arr = y_true.data.astype(np.float64, copy=True)
        diff = p_arr - y_arr
        batch_size = diff.shape[0]

        self._cache = (diff, y_pred.dtype)
        return float(np.sum(diff * diff) / batch_size)

    def backward(self) -> Tensor:
        """Trả về dL/dy_pred = (2 / N) * (y_pred - y_true) cùng dtype với y_pred."""
        if self._cache is None:
            raise StateError("FakeMSELoss: backward() phải được gọi sau forward().")
        diff, out_dtype = self._cache
        batch_size = diff.shape[0]
        grad = (2.0 / batch_size) * diff
        return Tensor(grad.astype(out_dtype, copy=False))
