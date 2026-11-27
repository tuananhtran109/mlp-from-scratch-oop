"""Lưu trữ và truy xuất lịch sử các chỉ số huấn luyện qua từng epoch."""

from __future__ import annotations

import numbers
from collections.abc import Mapping

import numpy as np

from nnlib.exceptions import InvalidConfigError, StateError


class TrainingHistory:
    """Lưu lịch sử loss/accuracy theo epoch (Vai trò: Thành phần dữ liệu của Trainer).

    Thuộc tính:
        _records (list[dict[str, float]]): Danh sách bản ghi chỉ số theo thứ tự epoch.
    """

    def __init__(self) -> None:
        """Khởi tạo lịch sử huấn luyện rỗng."""
        self._records: list[dict[str, float]] = []

    @property
    def records(self) -> list[dict[str, float]]:
        """Trả về bản sao danh sách các bản ghi chỉ số qua từng epoch."""
        return [dict(entry) for entry in self._records]

    def record(self, metrics: Mapping[str, numbers.Real]) -> None:
        """Ghi nhận bộ chỉ số của một epoch mới vào lịch sử."""
        if not isinstance(metrics, Mapping) or len(metrics) == 0:
            raise InvalidConfigError(
                "Chỉ số ghi nhận mỗi epoch phải là từ điển không rỗng."
            )

        clean_entry: dict[str, float] = {}
        for key, value in metrics.items():
            if not isinstance(key, str) or not key.strip():
                raise InvalidConfigError(
                    f"Tên chỉ số phải là chuỗi không rỗng, nhận được: {key!r}."
                )
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, numbers.Real
            ):
                raise InvalidConfigError(
                    f"Giá trị chỉ số '{key}' phải là số thực, nhận được: {value!r}."
                )
            clean_entry[key] = float(value)

        self._records.append(clean_entry)

    def get_metric(self, name: str) -> list[float]:
        """Trả về danh sách giá trị theo từng epoch của một chỉ số cụ thể."""
        if not isinstance(name, str) or not name.strip():
            raise InvalidConfigError(
                f"Tên chỉ số cần truy vấn phải là chuỗi không rỗng, nhận: {name!r}."
            )
        if not self._records:
            raise StateError("Lịch sử huấn luyện hiện đang trống.")

        values: list[float] = []
        for idx, entry in enumerate(self._records):
            if name not in entry:
                raise InvalidConfigError(
                    f"Không tìm thấy chỉ số '{name}' ở bản ghi epoch thứ {idx + 1}."
                )
            values.append(entry[name])
        return values

    def to_dict(self) -> dict[str, list[float]]:
        """Gom lịch sử thành từ điển {tên_chỉ_số: danh_sách_giá_trị}."""
        grouped: dict[str, list[float]] = {}
        for entry in self._records:
            for key, value in entry.items():
                grouped.setdefault(key, []).append(value)
        return grouped

    def latest(self) -> dict[str, float] | None:
        """Trả về bản sao chỉ số của epoch gần nhất hoặc None nếu chưa có."""
        if not self._records:
            return None
        return dict(self._records[-1])

    def clear(self) -> None:
        """Xóa toàn bộ bản ghi lịch sử hiện có."""
        self._records.clear()

    def __len__(self) -> int:
        """Trả về số epoch đã được ghi nhận."""
        return len(self._records)
