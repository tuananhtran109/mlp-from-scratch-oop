"""Bộ chia lô dữ liệu mini-batch và xáo trộn có kiểm soát seed (DataLoader)."""

from __future__ import annotations

from typing import Iterator

import numpy as np

from digitapp.data.dataset import IDataset
from digitapp.data.transforms import OneHotEncoder
from nnlib.exceptions import InvalidConfigError


class DataLoader:
    """Bộ lặp nạp dữ liệu mini-batch cho huấn luyện và đánh giá."""

    def __init__(
        self,
        dataset: IDataset,
        batch_size: int = 32,
        shuffle: bool = True,
        seed: int | None = None,
        one_hot: bool = True,
        num_classes: int = 10,
    ) -> None:
        """Khởi tạo DataLoader với tập dữ liệu, cỡ lô và cấu hình xáo trộn.

        Quy ước Rule R9:
            Bộ sinh số ngẫu nhiên _rng được khởi tạo một lần duy nhất và duy trì
            liên tục qua các epoch, bảo đảm mỗi epoch có thứ tự xáo trộn khác nhau
            nhưng vẫn tái lập được khi cung cấp cùng seed ban đầu.
        """
        # Rule R9
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, (int, np.integer))
            or int(batch_size) <= 0
        ):
            raise InvalidConfigError(
                f"batch_size phải là số nguyên dương, nhận được: {batch_size!r}."
            )

        self._dataset: IDataset = dataset
        self._batch_size: int = int(batch_size)
        self._shuffle: bool = bool(shuffle)
        self._one_hot: bool = bool(one_hot)
        self._encoder: OneHotEncoder | None = (
            OneHotEncoder(num_classes) if self._one_hot else None
        )

        # Duy trì liên tục qua các epoch huấn luyện
        self._rng: np.random.Generator = np.random.default_rng(seed)

    @property
    def batch_size(self) -> int:
        """Trả về kích thước mini-batch."""
        return self._batch_size

    @property
    def shuffle(self) -> bool:
        """Trả về trạng thái xáo trộn dữ liệu."""
        return self._shuffle

    @property
    def dataset(self) -> IDataset:
        """Trả về tập dữ liệu bên dưới."""
        return self._dataset

    def __len__(self) -> int:
        """Trả về tổng số batch trong 1 epoch (làm tròn lên để không mất mẫu dư)."""
        n_samples = len(self._dataset)
        if n_samples == 0:
            return 0
        return int(np.ceil(n_samples / self._batch_size))

    def __iter__(self) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Duyệt qua các batch trong một chu kỳ (epoch)."""
        n_samples = len(self._dataset)
        if n_samples == 0:
            return

        if self._shuffle:
            indices = self._rng.permutation(n_samples)
        else:
            indices = np.arange(n_samples)

        num_batches = len(self)
        for b in range(num_batches):
            start = b * self._batch_size
            end = min(start + self._batch_size, n_samples)
            batch_idx = indices[start:end]

            batch_x, batch_y = self._dataset[batch_idx]

            if self._encoder is not None:
                batch_y = self._encoder(batch_y)

            yield batch_x, batch_y
