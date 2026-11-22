"""Lớp trừu tượng cơ sở Layer cho toàn bộ các lớp mạng nơ-ron."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypeVar

import numpy as np

from nnlib.exceptions import InvalidConfigError, ShapeMismatchError, StateError
from nnlib.tensor import Parameter, Tensor

T = TypeVar("T")


class Layer(ABC):
    """Lớp cơ sở (Vai trò: Component trong mẫu Composite) cho mọi lớp mạng.

    Lưu ý cho lớp chứa lớp con (như ``Sequential``): cần ghi đè ``train()``,
    ``eval()`` để lan truyền trạng thái xuống các lớp con, và ghi đè
    ``state_dict()`` / ``load_state_dict()`` với tiền tố vị trí (vd ``0.weights``)
    để tránh trùng khóa giữa các lớp cùng loại.
    """

    def __init__(self, name: str | None = None) -> None:
        """Khởi tạo lớp với tên định danh và mặc định ở chế độ train."""
        self._name: str = name if name is not None else self.__class__.__name__
        self._training: bool = True

    @property
    def name(self) -> str:
        """Trả về tên của lớp mạng."""
        return self._name

    @property
    def training(self) -> bool:
        """Trả về True nếu lớp đang ở chế độ huấn luyện, False nếu eval."""
        return self._training

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """Lan truyền thuận từ Tensor đầu vào x sang Tensor đầu ra."""

    @abstractmethod
    def backward(self, grad_out: Tensor) -> Tensor:
        """Lan truyền ngược từ gradient đầu ra grad_out về đầu vào."""

    def parameters(self) -> list[Parameter]:
        """Trả về danh sách các Parameter học được của lớp."""
        return []

    def train(self) -> None:
        """Bật chế độ huấn luyện (ví dụ: Dropout hoạt động)."""
        self._training = True

    def eval(self) -> None:
        """Bật chế độ suy luận (ví dụ: Dropout trả nguyên đầu vào)."""
        self._training = False

    def _require_cache(self, cache: T | None) -> T:
        """Kiểm tra cache đã tồn tại trước khi chạy backward và trả về cache."""
        # Rule R2
        if cache is None:
            raise StateError(f"{self._name}: backward() phải được gọi sau forward().")
        return cache

    def state_dict(self) -> dict[str, np.ndarray]:
        """Trích xuất bản sao trọng số của lớp dưới dạng từ điển tên -> mảng."""
        state: dict[str, np.ndarray] = {}
        for idx, param in enumerate(self.parameters()):
            key = param.name if param.name else f"param_{idx}"
            state[key] = param.data.copy()
        return state

    def load_state_dict(self, state: dict[str, np.ndarray]) -> None:
        """Nạp trọng số từ từ điển vào lớp theo nguyên tắc toàn bộ hoặc không."""
        # Rule R6
        params = self.parameters()
        keys = [param.name or f"param_{i}" for i, param in enumerate(params)]
        validated: list[np.ndarray] = []

        for key, param in zip(keys, params, strict=True):
            if key not in state:
                raise InvalidConfigError(
                    f"Thiếu khóa tham số '{key}' cho lớp '{self._name}'."
                )
            arr = np.asarray(state[key])
            if arr.shape != param.shape:
                raise ShapeMismatchError(
                    f"Khóa '{key}': shape {arr.shape} khác {param.shape}."
                )
            validated.append(arr)

        for param, arr in zip(params, validated, strict=True):
            param.data = arr.copy()
