"""Tạo các tệp nhị phân IDX mẫu chuẩn (10 ảnh, 10 nhãn) phục vụ kiểm thử tự động."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np


def create_idx_samples(output_dir: Path) -> tuple[Path, Path]:
    """Tạo 2 tệp IDX mẫu chuẩn (t10k-images-idx3-ubyte và t10k-labels-idx1-ubyte)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    images_path = output_dir / "t10k-images-idx3-ubyte"
    labels_path = output_dir / "t10k-labels-idx1-ubyte"

    num_samples = 10
    rows = 28
    cols = 28

    # 1. Tạo tệp ảnh IDX3: magic 2051 (0x00000803), big-endian (>IIII)
    # Dùng seed cố định để dữ liệu tất định 100%
    rng = np.random.default_rng(42)
    images_data = rng.integers(0, 256, size=(num_samples, rows, cols), dtype=np.uint8)

    with open(images_path, "wb") as f:
        # Header: magic (2051), num_images (10), rows (28), cols (28)
        header = struct.pack(">IIII", 2051, num_samples, rows, cols)
        f.write(header)
        f.write(images_data.tobytes())

    # 2. Tạo tệp nhãn IDX1: magic 2049 (0x00000801), big-endian (>II)
    # Phủ đủ 10 chữ số từ 0 đến 9
    labels_data = np.arange(num_samples, dtype=np.uint8)

    with open(labels_path, "wb") as f:
        # Header: magic (2049), num_items (10)
        header = struct.pack(">II", 2049, num_samples)
        f.write(header)
        f.write(labels_data.tobytes())

    return images_path, labels_path


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    target_dir = project_root / "tests" / "fixtures" / "idx_sample"
    img_p, lbl_p = create_idx_samples(target_dir)
    print(f"Đã tạo tệp mẫu ảnh: {img_p} ({img_p.stat().st_size} bytes)")
    print(f"Đã tạo tệp mẫu nhãn: {lbl_p} ({lbl_p.stat().st_size} bytes)")
