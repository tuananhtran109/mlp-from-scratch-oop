"""Công cụ kiểm tra đạo hàm giải tích bằng sai phân trung tâm (float64)."""

from __future__ import annotations

import math
import numbers

import numpy as np

from nnlib.exceptions import InvalidConfigError
from nnlib.layers.base import Layer
from nnlib.losses.base import ILossFunction
from nnlib.tensor import Tensor


class GradientChecker:
    """So sánh gradient giải tích với gradient số học bằng sai phân trung tâm."""

    def __init__(
        self,
        epsilon: numbers.Real = 1e-5,
        tolerance: numbers.Real = 1e-5,
    ) -> None:
        """Khởi tạo GradientChecker với bước sai phân epsilon và ngưỡng tolerance."""
        self._epsilon: float = self._validate_positive_real(epsilon, "epsilon")
        self._tolerance: float = self._validate_positive_real(tolerance, "tolerance")

    @staticmethod
    def _validate_positive_real(value: numbers.Real, name: str) -> float:
        """Kiểm tra tham số số thực dương hữu hạn (loại bỏ bool)."""
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
            raise InvalidConfigError(
                f"Tham số {name} phải là số thực dương, nhận được {value!r}."
            )
        val = float(value)
        if not math.isfinite(val) or val <= 0.0:
            raise InvalidConfigError(
                f"Tham số {name} phải lớn hơn 0, nhận được {value!r}."
            )
        return val

    @property
    def epsilon(self) -> float:
        """Trả về bước nhiễu sai phân trung tâm epsilon."""
        return self._epsilon

    @property
    def tolerance(self) -> float:
        """Trả về ngưỡng sai số tương đối chấp nhận được."""
        return self._tolerance

    @staticmethod
    def relative_error(
        analytical: np.ndarray,
        numerical: np.ndarray,
    ) -> float:
        """Tính sai số tương đối cực đại |a - n| / max(|a| + |n|, 1e-12)."""
        if analytical.size == 0 and numerical.size == 0:
            return 0.0
        diff = np.abs(analytical - numerical)
        denom = np.maximum(np.abs(analytical) + np.abs(numerical), 1e-12)
        return float(np.max(diff / denom))

    @staticmethod
    def _require_float64(tensor: Tensor, label: str) -> None:
        """Bắt buộc dữ liệu dùng float64 khi kiểm tra gradient."""
        if tensor.dtype != np.float64:
            raise InvalidConfigError(
                f"GradientChecker yêu cầu {label} có dtype float64, "
                f"nhận được {tensor.dtype}."
            )

    def is_valid(self, errors: float | dict[str, float]) -> bool:
        """Kiểm tra toàn bộ sai số tương đối đều nhỏ hơn tolerance."""
        if isinstance(errors, dict):
            return all(err < self._tolerance for err in errors.values())
        return errors < self._tolerance

    def check_loss(
        self,
        loss_fn: ILossFunction,
        y_pred: Tensor,
        y_true: Tensor,
    ) -> float:
        """Kiểm tra gradient dL/dy_pred của một hàm mất mát."""
        self._require_float64(y_pred, "y_pred")
        _ = loss_fn.forward(y_pred, y_true)
        grad_ana = loss_fn.backward().data.copy()

        y_work = y_pred.data.copy()
        grad_num = np.zeros_like(y_work, dtype=np.float64)
        eps = self._epsilon

        for idx in np.ndindex(y_work.shape):
            orig = float(y_work[idx])
            y_work[idx] = orig + eps
            loss_pos = loss_fn.forward(Tensor(y_work, dtype=np.float64), y_true)
            y_work[idx] = orig - eps
            loss_neg = loss_fn.forward(Tensor(y_work, dtype=np.float64), y_true)
            y_work[idx] = orig
            grad_num[idx] = (loss_pos - loss_neg) / (2.0 * eps)

        return self.relative_error(grad_ana, grad_num)

    def check_input(
        self,
        layer: Layer,
        loss_fn: ILossFunction,
        x: Tensor,
        y_true: Tensor,
    ) -> float:
        """Kiểm tra gradient dL/dx truyền ngược qua một Layer hoặc mô hình."""
        self._require_float64(x, "x")
        for param in layer.parameters():
            self._require_float64(param, f"tham số '{param.name}'")
            param.zero_grad()

        y_pred = layer.forward(x)
        _ = loss_fn.forward(y_pred, y_true)
        grad_loss = loss_fn.backward()
        grad_x_ana = layer.backward(grad_loss).data.copy()

        x_work = x.data.copy()
        grad_x_num = np.zeros_like(x_work, dtype=np.float64)
        eps = self._epsilon

        for idx in np.ndindex(x_work.shape):
            orig = float(x_work[idx])
            x_work[idx] = orig + eps
            loss_pos = loss_fn.forward(
                layer.forward(Tensor(x_work, dtype=np.float64)), y_true
            )
            x_work[idx] = orig - eps
            loss_neg = loss_fn.forward(
                layer.forward(Tensor(x_work, dtype=np.float64)), y_true
            )
            x_work[idx] = orig
            grad_x_num[idx] = (loss_pos - loss_neg) / (2.0 * eps)

        return self.relative_error(grad_x_ana, grad_x_num)

    def check_parameters(
        self,
        model: Layer,
        loss_fn: ILossFunction,
        x: Tensor,
        y_true: Tensor,
    ) -> dict[str, float]:
        """Kiểm tra gradient giải tích của mọi Parameter trong model.

        Lưu ý: Nên đặt mô hình ở chế độ ``eval()`` nếu có lớp ngẫu nhiên
        như ``Dropout`` để phép sai phân trung tâm có tính tất định.
        """
        self._require_float64(x, "x")
        params = model.parameters()
        for param in params:
            self._require_float64(param, f"tham số '{param.name}'")
            param.zero_grad()

        y_pred = model.forward(x)
        _ = loss_fn.forward(y_pred, y_true)
        grad_loss = loss_fn.backward()
        _ = model.backward(grad_loss)

        analytical_grads = [param.grad.copy() for param in params]
        errors: dict[str, float] = {}
        eps = self._epsilon

        for p_idx, (param, grad_ana) in enumerate(
            zip(params, analytical_grads, strict=True)
        ):
            grad_num = np.zeros_like(param.data, dtype=np.float64)
            for idx in np.ndindex(param.shape):
                orig = float(param.data[idx])
                try:
                    param.data[idx] = orig + eps
                    loss_pos = loss_fn.forward(model.forward(x), y_true)
                    param.data[idx] = orig - eps
                    loss_neg = loss_fn.forward(model.forward(x), y_true)
                finally:
                    param.data[idx] = orig
                grad_num[idx] = (loss_pos - loss_neg) / (2.0 * eps)

            key = f"{p_idx}.{param.name}" if param.name else f"{p_idx}.param"
            errors[key] = self.relative_error(grad_ana, grad_num)

        return errors
