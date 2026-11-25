"""Định nghĩa giao diện IDataset và lớp MnistDataset nạp dữ liệu MNIST IDX."""

from __future__ import annotations

import gzip
import struct
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from digitapp.common.paths import resource_path
from digitapp.exceptions import DataFormatError
from nnlib.exceptions import InvalidConfigError


class IDataset(ABC):
    """Giao diện trừu tượng cho tập dữ liệu."""

    @abstractmethod
    def __len__(self) -> int:
        """Trả về tổng số mẫu trong tập dữ liệu."""

    @abstractmethod
    def __getitem__(
        self, idx: int | slice | np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Truy xuất một hoặc nhiều mẫu dữ liệu (ảnh và nhãn)."""


class MnistDataset(IDataset):
    """Tập dữ liệu chữ số MNIST nạp từ định dạng nhị phân IDX."""

    def __init__(self, images: np.ndarray, labels: np.ndarray) -> None:
        """Khởi tạo MnistDataset từ mảng ảnh float32 và nhãn số nguyên int64."""
        if images.ndim != 2 or images.shape[1] != 784:
            raise InvalidConfigError(
                f"Ảnh phải có số chiều (N, 784), nhận được: {images.shape}."
            )
        if labels.ndim != 1 or len(images) != len(labels):
            raise InvalidConfigError(
                f"Số lượng ảnh ({len(images)}) và nhãn ({len(labels)}) "
                f"phải khớp nhau."
            )

        self._images: np.ndarray = np.asarray(images, dtype=np.float32)
        self._labels: np.ndarray = np.asarray(labels, dtype=np.int64)

    @property
    def images(self) -> np.ndarray:
        """Trả về mảng ảnh chuẩn hóa float32 shape (N, 784)."""
        return self._images

    @property
    def labels(self) -> np.ndarray:
        """Trả về mảng nhãn số nguyên int64 shape (N,)."""
        return self._labels

    def __len__(self) -> int:
        """Trả về số lượng mẫu trong tập dữ liệu."""
        return len(self._images)

    def __getitem__(
        self, idx: int | slice | np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Lấy mẫu hoặc tập con theo chỉ số."""
        return self._images[idx], self._labels[idx]

    @classmethod
    def from_idx(
        cls,
        images_path: str | Path,
        labels_path: str | Path,
    ) -> MnistDataset:
        """Nạp tập dữ liệu từ cặp tệp nhị phân IDX (hỗ trợ cả tệp nén .gz)."""
        img_p = resource_path(images_path)
        lbl_p = resource_path(labels_path)

        images = cls._read_idx_images(img_p)
        labels = cls._read_idx_labels(lbl_p)

        if len(images) != len(labels):
            raise DataFormatError(
                f"Số lượng ảnh ({len(images)}) không khớp với số lượng nhãn "
                f"({len(labels)})."
            )

        return cls(images, labels)

    @staticmethod
    def _open_file(path: Path):
        """Mở tệp thông thường hoặc tệp gzip tùy theo phần mở rộng."""
        if not path.is_file():
            raise DataFormatError(f"Không tìm thấy tệp dữ liệu tại: {path}")

        if path.suffix == ".gz":
            return gzip.open(path, "rb")
        return open(path, "rb")

    @classmethod
    def _read_idx_images(cls, path: Path) -> np.ndarray:
        """Đọc tệp ảnh IDX3: magic 2051 (0x00000803), chuẩn hóa chia 255 về float32."""
        try:
            with cls._open_file(path) as f:
                header_bytes = f.read(16)
                if len(header_bytes) < 16:
                    raise DataFormatError(
                        f"Tệp ảnh {path.name} bị cụt, header không đủ 16 byte."
                    )

                magic, num_images, rows, cols = struct.unpack(">IIII", header_bytes)
                if magic != 2051:
                    raise DataFormatError(
                        f"Magic number tệp ảnh không đúng: kỳ vọng 2051 (0x00000803), "
                        f"nhận được {magic}."
                    )
                if rows != 28 or cols != 28:
                    raise DataFormatError(
                        f"Kích thước ảnh không đúng: kỳ vọng 28x28, "
                        f"nhận được {rows}x{cols}."
                    )

                expected_bytes = num_images * rows * cols
                raw_data = f.read(expected_bytes)
                if len(raw_data) < expected_bytes:
                    raise DataFormatError(
                        f"Tệp ảnh {path.name} bị thiếu dữ liệu pixel: "
                        f"kỳ vọng {expected_bytes} bytes, nhận được {len(raw_data)}."
                    )

                # Chuyển đổi sang mảng uint8, chia 255.0 duy nhất một lần về float32
                arr = np.frombuffer(raw_data, dtype=np.uint8).astype(np.float32)
                arr /= 255.0
                return arr.reshape((num_images, 784))
        except (OSError, struct.error) as exc:
            raise DataFormatError(
                f"Lỗi khi đọc tệp ảnh IDX {path.name}: {exc}"
            ) from exc

    @classmethod
    def _read_idx_labels(cls, path: Path) -> np.ndarray:
        """Đọc tệp nhãn IDX1: magic 2049 (0x00000801), lưu nhãn số nguyên int64."""
        try:
            with cls._open_file(path) as f:
                header_bytes = f.read(8)
                if len(header_bytes) < 8:
                    raise DataFormatError(
                        f"Tệp nhãn {path.name} bị cụt, header không đủ 8 byte."
                    )

                magic, num_items = struct.unpack(">II", header_bytes)
                if magic != 2049:
                    raise DataFormatError(
                        f"Magic number tệp nhãn không đúng: kỳ vọng 2049 (0x00000801), "
                        f"nhận được {magic}."
                    )

                raw_data = f.read(num_items)
                if len(raw_data) < num_items:
                    raise DataFormatError(
                        f"Tệp nhãn {path.name} bị thiếu dữ liệu: "
                        f"kỳ vọng {num_items} bytes, nhận được {len(raw_data)}."
                    )

                labels = np.frombuffer(raw_data, dtype=np.uint8).astype(np.int64)
                return labels
        except (OSError, struct.error) as exc:
            raise DataFormatError(
                f"Lỗi khi đọc tệp nhãn IDX {path.name}: {exc}"
            ) from exc

    def split(
        self,
        val_ratio: float,
        seed: int = 42,
    ) -> tuple[MnistDataset, MnistDataset]:
        """Tách tập dữ liệu thành train và validation (Rule R9).

        Quy tắc R9:
            - Tách train/val theo tỉ lệ cố định với seed xác định.
            - Hai tập không giao nhau, hợp lại đủ N mẫu.
            - Đảm bảo mỗi tập chứa ít nhất 1 mẫu.
        """
        # Rule R9
        if (
            isinstance(val_ratio, bool)
            or not isinstance(val_ratio, (int, float))
            or np.isnan(val_ratio)
            or not (0.0 < float(val_ratio) < 1.0)
        ):
            raise InvalidConfigError(
                f"Tỉ lệ val_ratio phải là số thực trong khoảng (0, 1), "
                f"nhận được: {val_ratio!r}."
            )

        n_total = len(self._images)
        if n_total < 2:
            raise InvalidConfigError("Không thể chia tập dữ liệu có ít hơn 2 mẫu.")

        n_val = int(round(n_total * float(val_ratio)))
        # Đảm bảo mỗi tập có ít nhất 1 mẫu
        n_val = max(1, min(n_total - 1, n_val))

        rng = np.random.default_rng(seed)
        perm = rng.permutation(n_total)

        val_indices = perm[:n_val]
        train_indices = perm[n_val:]

        train_set = MnistDataset(
            self._images[train_indices], self._labels[train_indices]
        )
        val_set = MnistDataset(self._images[val_indices], self._labels[val_indices])

        return train_set, val_set
