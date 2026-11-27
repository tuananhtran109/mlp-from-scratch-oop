"""Thuật toán tối ưu hóa Adam (Adaptive Moment Estimation) có bias correction."""

from __future__ import annotations

import math
import numbers
from typing import Sequence

import numpy as np

from nnlib.exceptions import InvalidConfigError, ShapeMismatchError
from nnlib.optim.base import IOptimizer
from nnlib.tensor import Parameter


class Adam(IOptimizer):
    """Bộ tối ưu hóa Adam (Vai trò: Concrete Strategy trong mẫu Strategy).

    Thuộc tính:
        _lr (float): Tốc độ học dương (> 0), kế thừa từ ``IOptimizer``.
        _beta1 (float): Hệ số suy giảm cho ước lượng moment bậc 1 trong [0, 1).
        _beta2 (float): Hệ số suy giảm cho ước lượng moment bậc 2 trong [0, 1).
        _eps (float): Hằng số ổn định số dương (> 0) ở mẫu số cập nhật.
        _m (dict[int, np.ndarray]): Moment bậc 1 theo chỉ số tham số (cùng dtype).
        _v (dict[int, np.ndarray]): Moment bậc 2 theo chỉ số tham số (cùng dtype).
        _t (int): Bộ đếm bước thời gian phục vụ hiệu chỉnh độ lệch (bias correction).
    """

    def __init__(
        self,
        lr: numbers.Real = 1e-3,
        beta1: numbers.Real = 0.9,
        beta2: numbers.Real = 0.999,
        eps: numbers.Real = 1e-8,
    ) -> None:
        """Khởi tạo Adam với các siêu tham số mặc định theo đặc tả mục 7.

        Quy tắc R7:
            - ``lr > 0`` và ``eps > 0`` (hữu hạn, loại trừ kiểu boolean).
            - ``beta1``, ``beta2`` thuộc nửa khoảng ``[0, 1)`` (loại trừ boolean).
        """
        # Rule R7
        if isinstance(lr, (bool, np.bool_)):
            raise InvalidConfigError(
                f"Tốc độ học (lr) phải là số thực dương, nhận được: {lr!r}."
            )
        normalized_lr = (
            float(lr) if isinstance(lr, numbers.Real) else lr  # type: ignore[arg-type]
        )
        super().__init__(lr=normalized_lr)

        self._beta1: float = self._validate_beta(beta1, "beta1")
        self._beta2: float = self._validate_beta(beta2, "beta2")
        self._eps: float = self._validate_positive_real(eps, "eps")

        # Lưu moment theo chỉ số nguyên (index) của tham số, không dùng id()
        self._m: dict[int, np.ndarray] = {}
        self._v: dict[int, np.ndarray] = {}
        self._t: int = 0

    @staticmethod
    def _validate_beta(value: numbers.Real, name: str) -> float:
        """Kiểm tra hệ số beta thuộc [0, 1), hữu hạn và không phải boolean."""
        # Rule R7
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
            raise InvalidConfigError(
                f"Hệ số {name} phải là số thực trong [0, 1), nhận được: {value!r}."
            )
        val = float(value)
        if not math.isfinite(val) or not (0.0 <= val < 1.0):
            raise InvalidConfigError(
                f"Hệ số {name} phải thuộc khoảng [0, 1), nhận được: {value!r}."
            )
        return val

    @staticmethod
    def _validate_positive_real(value: numbers.Real, name: str) -> float:
        """Kiểm tra tham số thực dương hữu hạn và không phải boolean."""
        # Rule R7
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
            raise InvalidConfigError(
                f"Tham số {name} phải là số thực dương, nhận được: {value!r}."
            )
        val = float(value)
        if not math.isfinite(val) or val <= 0.0:
            raise InvalidConfigError(
                f"Tham số {name} phải là số thực dương (> 0), nhận được: {value!r}."
            )
        return val

    @property
    def beta1(self) -> float:
        """Trả về hệ số suy giảm moment bậc 1 (beta1)."""
        return self._beta1

    @property
    def beta2(self) -> float:
        """Trả về hệ số suy giảm moment bậc 2 (beta2)."""
        return self._beta2

    @property
    def eps(self) -> float:
        """Trả về hằng số ổn định số epsilon."""
        return self._eps

    @property
    def t(self) -> int:
        """Trả về số bước cập nhật thời gian hiện tại."""
        return self._t

    @property
    def m(self) -> dict[int, np.ndarray]:
        """Trả về bản sao từ điển trạng thái moment bậc 1 theo chỉ số tham số."""
        return {idx: arr.copy() for idx, arr in self._m.items()}

    @property
    def v(self) -> dict[int, np.ndarray]:
        """Trả về bản sao từ điển trạng thái moment bậc 2 theo chỉ số tham số."""
        return {idx: arr.copy() for idx, arr in self._v.items()}

    def step(self, params: Sequence[Parameter]) -> None:
        """Cập nhật tham số theo công thức Adam có hiệu chỉnh độ lệch (bias correction).

        Công thức:
            m <- beta1 * m + (1 - beta1) * g
            v <- beta2 * v + (1 - beta2) * g^2
            m_hat = m / (1 - beta1^t)
            v_hat = v / (1 - beta2^t)
            theta <- theta - lr * m_hat / (sqrt(v_hat) + eps)
        """
        # Rule R7
        if len(params) == 0:
            return

        for idx, param in enumerate(params):
            if idx in self._m:
                if self._m[idx].shape != param.data.shape:
                    raise ShapeMismatchError(
                        f"Tham số tại vị trí {idx} có shape {param.data.shape} "
                        f"không khớp trạng thái Adam shape {self._m[idx].shape}."
                    )
                if self._m[idx].dtype != param.data.dtype:
                    raise InvalidConfigError(
                        f"Tham số tại vị trí {idx} có dtype {param.data.dtype} "
                        f"không khớp trạng thái Adam dtype {self._m[idx].dtype}."
                    )

        self._t += 1
        bias_corr1 = 1.0 - (self._beta1**self._t)
        bias_corr2 = 1.0 - (self._beta2**self._t)

        for idx, param in enumerate(params):
            dtype = param.data.dtype
            if idx not in self._m:
                self._m[idx] = np.zeros_like(param.data)
                self._v[idx] = np.zeros_like(param.data)

            grad = param.grad.astype(dtype, copy=False)
            b1 = dtype.type(self._beta1)
            b2 = dtype.type(self._beta2)
            one_minus_b1 = dtype.type(1.0 - self._beta1)
            one_minus_b2 = dtype.type(1.0 - self._beta2)

            m_new = b1 * self._m[idx] + one_minus_b1 * grad
            v_new = b2 * self._v[idx] + one_minus_b2 * (grad * grad)
            self._m[idx] = m_new
            self._v[idx] = v_new

            m_hat = m_new / dtype.type(bias_corr1)
            v_hat = v_new / dtype.type(bias_corr2)
            denom = np.sqrt(v_hat) + dtype.type(self._eps)
            step_update = dtype.type(self._lr) * m_hat / denom

            param.data = param.data - step_update
