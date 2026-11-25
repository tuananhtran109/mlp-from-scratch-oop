"""Kiểm thử đơn vị cho MnistDataset và cơ chế split (Chỉ tiêu T19, Quy tắc R9)."""

import gzip
import struct
from pathlib import Path

import numpy as np
import pytest

from digitapp.common.paths import resource_path
from digitapp.data.dataset import MnistDataset
from digitapp.exceptions import DataFormatError
from nnlib.exceptions import InvalidConfigError


def test_mnist_dataset_from_fixture_t19() -> None:
    """Kiểm thử T19: Đọc thành công cặp tệp IDX mẫu chuẩn."""
    img_path = resource_path("tests/fixtures/idx_sample/t10k-images-idx3-ubyte")
    lbl_path = resource_path("tests/fixtures/idx_sample/t10k-labels-idx1-ubyte")

    dataset = MnistDataset.from_idx(img_path, lbl_path)

    # 1. Kiểm tra kích thước và kiểu dữ liệu
    assert len(dataset) == 10
    assert dataset.images.shape == (10, 784)
    assert dataset.images.dtype == np.float32
    assert dataset.labels.shape == (10,)
    assert dataset.labels.dtype == np.int64

    # 2. Kiểm tra khoảng giá trị chuẩn hóa [0.0, 1.0]
    assert dataset.images.min() >= 0.0
    assert dataset.images.max() <= 1.0

    # 3. Kiểm tra nhãn phủ đủ 0..9
    assert set(dataset.labels.tolist()) == set(range(10))

    # 4. Kiểm tra __getitem__
    img0, lbl0 = dataset[0]
    assert img0.shape == (784,)
    assert lbl0 == 0

    sub_imgs, sub_lbls = dataset[0:3]
    assert sub_imgs.shape == (3, 784)
    assert sub_lbls.shape == (3,)


def test_mnist_dataset_from_gzip(tmp_path: Path) -> None:
    """Kiểm thử đọc tệp nén .gz tự động."""
    img_path = resource_path("tests/fixtures/idx_sample/t10k-images-idx3-ubyte")
    lbl_path = resource_path("tests/fixtures/idx_sample/t10k-labels-idx1-ubyte")

    gz_img = tmp_path / "images.gz"
    gz_lbl = tmp_path / "labels.gz"

    with open(img_path, "rb") as f_in, gzip.open(gz_img, "wb") as f_out:
        f_out.write(f_in.read())
    with open(lbl_path, "rb") as f_in, gzip.open(gz_lbl, "wb") as f_out:
        f_out.write(f_in.read())

    dataset = MnistDataset.from_idx(gz_img, gz_lbl)
    assert len(dataset) == 10
    assert dataset.images.shape == (10, 784)


def test_mnist_dataset_invalid_magic_t19(tmp_path: Path) -> None:
    """Kiểm thử T19: Magic number sai ném DataFormatError."""
    bad_img = tmp_path / "bad_magic_img"
    bad_lbl = tmp_path / "bad_magic_lbl"

    # Header ảnh với magic 9999
    bad_img.write_bytes(struct.pack(">IIII", 9999, 10, 28, 28) + b"\x00" * 7840)
    bad_lbl.write_bytes(struct.pack(">II", 2049, 10) + b"\x00" * 10)

    with pytest.raises(DataFormatError) as exc_info:
        MnistDataset.from_idx(bad_img, bad_lbl)
    assert "Magic number tệp ảnh không đúng" in str(exc_info.value)

    # Header nhãn với magic 8888
    good_img = tmp_path / "good_img"
    good_img.write_bytes(struct.pack(">IIII", 2051, 10, 28, 28) + b"\x00" * 7840)
    bad_lbl.write_bytes(struct.pack(">II", 8888, 10) + b"\x00" * 10)

    with pytest.raises(DataFormatError) as exc_info:
        MnistDataset.from_idx(good_img, bad_lbl)
    assert "Magic number tệp nhãn không đúng" in str(exc_info.value)


def test_mnist_dataset_truncated_file_t19(tmp_path: Path) -> None:
    """Kiểm thử T19: Tệp bị cụt header hoặc thiếu byte ném DataFormatError."""
    trunc_img = tmp_path / "trunc_img"
    trunc_lbl = tmp_path / "trunc_lbl"

    # Chỉ ghi 10 byte header thay vì 16
    trunc_img.write_bytes(b"\x00" * 10)
    trunc_lbl.write_bytes(struct.pack(">II", 2049, 10) + b"\x00" * 10)

    with pytest.raises(DataFormatError) as exc_info:
        MnistDataset.from_idx(trunc_img, trunc_lbl)
    assert "bị cụt, header không đủ 16 byte" in str(exc_info.value)

    # Header hợp lệ 10 ảnh nhưng chỉ ghi 500 byte dữ liệu thay vì 7840
    trunc_img.write_bytes(struct.pack(">IIII", 2051, 10, 28, 28) + b"\x00" * 500)
    with pytest.raises(DataFormatError) as exc_info:
        MnistDataset.from_idx(trunc_img, trunc_lbl)
    assert "bị thiếu dữ liệu pixel" in str(exc_info.value)


def test_mnist_dataset_mismatched_counts_t19(tmp_path: Path) -> None:
    """Kiểm thử T19: Số lượng ảnh và nhãn không bằng nhau ném DataFormatError."""
    img_file = tmp_path / "img_10"
    lbl_file = tmp_path / "lbl_5"

    img_file.write_bytes(struct.pack(">IIII", 2051, 10, 28, 28) + b"\x00" * 7840)
    lbl_file.write_bytes(struct.pack(">II", 2049, 5) + b"\x00" * 5)

    with pytest.raises(DataFormatError) as exc_info:
        MnistDataset.from_idx(img_file, lbl_file)
    assert "không khớp" in str(exc_info.value)


def test_mnist_dataset_split_rule_r9() -> None:
    """Kiểm thử Rule R9: Tách train/val không giao nhau, hợp đủ N, seed tái lập."""
    img_path = resource_path("tests/fixtures/idx_sample/t10k-images-idx3-ubyte")
    lbl_path = resource_path("tests/fixtures/idx_sample/t10k-labels-idx1-ubyte")
    dataset = MnistDataset.from_idx(img_path, lbl_path)

    train_set, val_set = dataset.split(val_ratio=0.2, seed=42)

    # 1. Đúng số lượng mẫu: 10 * 0.2 = 2 val, 8 train
    assert len(train_set) == 8
    assert len(val_set) == 2
    assert len(train_set) + len(val_set) == len(dataset)

    # 2. Hai tập không giao nhau và hợp đủ tập ban đầu
    train_labels = set(train_set.labels.tolist())
    val_labels = set(val_set.labels.tolist())
    assert len(train_labels.intersection(val_labels)) == 0
    assert train_labels.union(val_labels) == set(range(10))

    # 3. Cùng seed sinh ra phân tách y hệt
    train_2, val_2 = dataset.split(val_ratio=0.2, seed=42)
    assert np.array_equal(train_set.labels, train_2.labels)
    assert np.array_equal(val_set.labels, val_2.labels)

    # 4. Khác seed sinh ra phân tách khác
    _, val_3 = dataset.split(val_ratio=0.2, seed=123)
    assert not np.array_equal(val_set.labels, val_3.labels)


@pytest.mark.parametrize("invalid_ratio", [0.0, 1.0, -0.1, 1.5, True])
def test_mnist_dataset_split_invalid_ratio(invalid_ratio: object) -> None:
    """val_ratio nằm ngoài (0, 1) hoặc là bool phải ném InvalidConfigError."""
    dataset = MnistDataset(
        np.zeros((10, 784), dtype=np.float32), np.zeros(10, dtype=np.int64)
    )
    with pytest.raises(InvalidConfigError):
        dataset.split(val_ratio=invalid_ratio)  # type: ignore[arg-type]


def test_mnist_dataset_split_small_dataset_clamping() -> None:
    """Tập nhỏ (2 mẫu) với val_ratio rất nhỏ vẫn đảm bảo mỗi tập >= 1 mẫu."""
    dataset = MnistDataset(
        np.zeros((2, 784), dtype=np.float32), np.array([0, 1], dtype=np.int64)
    )
    train_set, val_set = dataset.split(val_ratio=0.01, seed=42)
    assert len(train_set) == 1
    assert len(val_set) == 1
