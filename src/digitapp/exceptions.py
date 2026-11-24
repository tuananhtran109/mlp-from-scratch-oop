"""Hệ thống ngoại lệ đặc thù cho phân hệ ứng dụng digitapp (Người B)."""

from __future__ import annotations

from nnlib.exceptions import NNError


class AppError(NNError):
    """Ngoại lệ cơ sở cho toàn bộ tầng ứng dụng digitapp."""


class DataFormatError(AppError):
    """Ném ra khi tệp dữ liệu IDX hoặc ảnh có định dạng không hợp lệ hoặc bị hỏng."""


class EmptyCanvasError(AppError):
    """Ném ra khi người dùng yêu cầu nhận diện nhưng khung vẽ canvas trống."""
