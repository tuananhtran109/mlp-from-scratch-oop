"""Kiểm thử đơn vị cho cây ngoại lệ phân hệ ứng dụng digitapp (Người B)."""

import pytest

from digitapp.exceptions import AppError, DataFormatError, EmptyCanvasError
from nnlib.exceptions import NNError


def test_app_exceptions_hierarchy() -> None:
    """Các ngoại lệ của B kế thừa AppError và NNError, lưu giữ message."""
    for exc_cls in (AppError, DataFormatError, EmptyCanvasError):
        msg = f"Lỗi thử nghiệm của {exc_cls.__name__}"
        err = exc_cls(msg)

        assert isinstance(err, NNError)
        assert isinstance(err, AppError)
        assert err.message == msg
        assert str(err) == msg


def test_catch_app_exceptions_polymorphically() -> None:
    """Có thể bắt đa hình thông qua AppError hoặc NNError."""
    with pytest.raises(NNError):
        raise DataFormatError("Sai magic number IDX")

    with pytest.raises(AppError):
        raise EmptyCanvasError("Canvas chưa có nét vẽ")
