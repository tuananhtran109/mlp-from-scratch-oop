"""Quản lý đường dẫn tài nguyên an toàn cho môi trường chạy mã nguồn và PyInstaller."""

from __future__ import annotations

import sys
from pathlib import Path


def get_project_root() -> Path:
    """Trả về đường dẫn tuyệt đối tới thư mục gốc dự án."""
    # src/digitapp/common/paths.py -> 3 cấp thư mục cha trỏ về mlp-from-scratch-oop/
    return Path(__file__).resolve().parents[3]


def resource_path(relative_path: str | Path) -> Path:
    """Chuyển đổi đường dẫn tương đối thành đường dẫn tuyệt đối an toàn.

    Hỗ trợ cả môi trường phát triển mã nguồn thông thường và môi trường
    đóng gói một tệp thực thi bởi PyInstaller (thông qua sys._MEIPASS).
    """
    rel = Path(relative_path)
    if rel.is_absolute():
        return rel.resolve()

    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        # Ghi chú: Thư mục sys._MEIPASS là thư mục tạm chỉ đọc (read-only)
        # của PyInstaller khi giải nén tài nguyên kèm theo (--add-data).
        # Các dữ liệu cần ghi khi chạy ứng dụng (models/, logs/)
        # sẽ được cấu hình lưu ra thư mục bên ngoài tại tuần đóng gói.
        base_dir = Path(sys._MEIPASS)
    else:
        base_dir = get_project_root()

    return (base_dir / rel).resolve()
