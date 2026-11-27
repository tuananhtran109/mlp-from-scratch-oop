"""Giao diện ICallback (Observer) và lớp LoggingCallback theo dõi huấn luyện."""

from __future__ import annotations

import logging
from abc import ABC
from typing import TYPE_CHECKING, Any

import numpy as np

from nnlib.exceptions import InvalidConfigError

if TYPE_CHECKING:
    from nnlib.training.trainer import Trainer


class ICallback(ABC):
    """Giao diện theo dõi sự kiện huấn luyện (Vai trò: Observer trong mẫu Observer).

    Các phương thức sự kiện mặc định là rỗng để lớp con chỉ cần ghi đè những
    mốc thời điểm cần dùng.

    Thuộc tính:
        _trainer (Trainer | None): Tham chiếu tới đối tượng Trainer chủ quản.
    """

    def __init__(self) -> None:
        """Khởi tạo callback với tham chiếu Trainer ban đầu là None."""
        self._trainer: Trainer | None = None

    @property
    def trainer(self) -> Trainer | None:
        """Trả về đối tượng Trainer đang gắn với callback này."""
        return self._trainer

    def set_trainer(self, trainer: Trainer) -> None:
        """Gắn tham chiếu Trainer cho callback khi đăng ký với Subject."""
        self._trainer = trainer

    def on_train_begin(self, logs: dict[str, Any] | None = None) -> None:
        """Xử lý sự kiện ngay trước khi bắt đầu vòng lặp huấn luyện."""

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        """Xử lý sự kiện ngay sau khi kết thúc mỗi epoch huấn luyện."""

    def on_train_end(self, logs: dict[str, Any] | None = None) -> None:
        """Xử lý sự kiện ngay sau khi kết thúc toàn bộ quá trình huấn luyện."""


class LoggingCallback(ICallback):
    """Ghi nhật ký quá trình huấn luyện qua logging (Vai trò: Concrete Observer).

    Thuộc tính:
        _trainer (Trainer | None): Tham chiếu Trainer kế thừa từ ``ICallback``.
        _logger (logging.Logger): Bộ ghi nhật ký chuẩn của thư viện ``logging``.
        _level (int): Mức độ log (mặc định ``logging.INFO``).
    """

    def __init__(
        self,
        logger: logging.Logger | None = None,
        level: int = logging.INFO,
    ) -> None:
        """Khởi tạo LoggingCallback với logger và mức log tùy chọn."""
        super().__init__()
        if logger is not None and not isinstance(logger, logging.Logger):
            raise InvalidConfigError(
                f"logger phải là logging.Logger hoặc None, nhận được: {logger!r}."
            )
        if (
            isinstance(level, (bool, np.bool_))
            or not isinstance(level, (int, np.integer))
            or int(level) < 0
        ):
            raise InvalidConfigError(
                f"Mức log phải là số nguyên không âm, nhận được: {level!r}."
            )
        self._logger: logging.Logger = (
            logger if logger is not None else logging.getLogger("nnlib.training")
        )
        self._level: int = int(level)

    @property
    def logger(self) -> logging.Logger:
        """Trả về đối tượng Logger đang sử dụng."""
        return self._logger

    @property
    def level(self) -> int:
        """Trả về mức độ ghi log hiện tại."""
        return self._level

    @staticmethod
    def _format_logs(logs: dict[str, Any] | None) -> str:
        """Định dạng từ điển chỉ số thành chuỗi dễ đọc cho nhật ký."""
        if not logs:
            return ""
        parts: list[str] = []
        for key, value in logs.items():
            if isinstance(value, float):
                parts.append(f"{key}={value:.4f}")
            else:
                parts.append(f"{key}={value}")
        return ", ".join(parts)

    def on_train_begin(self, logs: dict[str, Any] | None = None) -> None:
        """Ghi log thời điểm bắt đầu huấn luyện."""
        summary = self._format_logs(logs)
        if summary:
            self._logger.log(self._level, "Bắt đầu huấn luyện (%s).", summary)
        else:
            self._logger.log(self._level, "Bắt đầu huấn luyện.")

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        """Ghi log kết quả loss và độ chính xác sau mỗi epoch."""
        summary = self._format_logs(logs)
        if summary:
            self._logger.log(self._level, "Epoch %d: %s", epoch, summary)
        else:
            self._logger.log(self._level, "Epoch %d hoàn tất.", epoch)

    def on_train_end(self, logs: dict[str, Any] | None = None) -> None:
        """Ghi log tổng kết khi quá trình huấn luyện kết thúc."""
        summary = self._format_logs(logs)
        if summary:
            self._logger.log(self._level, "Kết thúc huấn luyện (%s).", summary)
        else:
            self._logger.log(self._level, "Kết thúc huấn luyện.")
