"""Kiểm thử đơn vị cho DataLoader và OneHotEncoder (Chỉ tiêu T18, Quy tắc R9)."""

import numpy as np
import pytest

from digitapp.common.paths import resource_path
from digitapp.data.dataloader import DataLoader
from digitapp.data.dataset import MnistDataset
from digitapp.data.transforms import Normalizer, OneHotEncoder
from nnlib.exceptions import InvalidConfigError


@pytest.fixture
def sample_dataset() -> MnistDataset:
    """Nạp dataset mẫu 10 phần tử từ fixture."""
    img_path = resource_path("tests/fixtures/idx_sample/t10k-images-idx3-ubyte")
    lbl_path = resource_path("tests/fixtures/idx_sample/t10k-labels-idx1-ubyte")
    return MnistDataset.from_idx(img_path, lbl_path)


def test_dataloader_batch_size_and_remainder_t18(
    sample_dataset: MnistDataset,
) -> None:
    """Kiểm thử T18: Chia lô chính xác và giữ trọn vẹn mẫu ở batch cuối cùng."""
    # 10 mẫu chia batch_size=4 -> 3 batch kích thước [4, 4, 2]
    loader = DataLoader(sample_dataset, batch_size=4, shuffle=False)
    assert len(loader) == 3

    batches = list(loader)
    assert len(batches) == 3

    # Batch 1
    bx1, by1 = batches[0]
    assert bx1.shape == (4, 784)
    assert by1.shape == (4, 10)

    # Batch 2
    bx2, by2 = batches[1]
    assert bx2.shape == (4, 784)
    assert by2.shape == (4, 10)

    # Batch 3 (phần dư)
    bx3, by3 = batches[2]
    assert bx3.shape == (2, 784)
    assert by3.shape == (2, 10)

    total_samples = sum(len(bx) for bx, _ in batches)
    assert total_samples == 10


def test_dataloader_one_hot_encoding_t18(sample_dataset: MnistDataset) -> None:
    """Kiểm thử T18: Chuyển đổi nhãn dạng one-hot theo tùy chọn."""
    # 1. one_hot=True (mặc định)
    loader_oh = DataLoader(sample_dataset, batch_size=10, shuffle=False, one_hot=True)
    _, by_oh = next(iter(loader_oh))
    assert by_oh.shape == (10, 10)
    assert by_oh.dtype == np.float32
    assert np.all(by_oh.sum(axis=1) == 1.0)
    # Nhãn 0..9
    for i in range(10):
        assert by_oh[i, i] == 1.0

    # 2. one_hot=False
    loader_int = DataLoader(sample_dataset, batch_size=10, shuffle=False, one_hot=False)
    _, by_int = next(iter(loader_int))
    assert by_int.shape == (10,)
    assert by_int.dtype == np.int64
    assert np.array_equal(by_int, np.arange(10))


def test_dataloader_shuffle_rng_continuity_rule_r9(
    sample_dataset: MnistDataset,
) -> None:
    """Kiểm thử Rule R9: rng duy trì liên tục qua các epoch; cùng seed tái lập."""
    loader1 = DataLoader(
        sample_dataset, batch_size=5, shuffle=True, seed=42, one_hot=False
    )

    # Lấy thứ tự các nhãn qua epoch 1
    epoch1_labels = []
    for _, by in loader1:
        epoch1_labels.extend(by.tolist())

    # Lấy thứ tự các nhãn qua epoch 2 (cùng loader1, rng duy trì liên tục)
    epoch2_labels = []
    for _, by in loader1:
        epoch2_labels.extend(by.tolist())

    # Thứ tự 2 epoch phải khác nhau vì rng tiếp tục sinh ngẫu nhiên
    assert epoch1_labels != epoch2_labels
    assert sorted(epoch1_labels) == list(range(10))
    assert sorted(epoch2_labels) == list(range(10))

    # Loader mới với cùng seed phải tái lập chính xác epoch 1
    loader2 = DataLoader(
        sample_dataset, batch_size=5, shuffle=True, seed=42, one_hot=False
    )
    reproduced_epoch1 = []
    for _, by in loader2:
        reproduced_epoch1.extend(by.tolist())

    assert epoch1_labels == reproduced_epoch1


def test_dataloader_no_shuffle(sample_dataset: MnistDataset) -> None:
    """Khi shuffle=False, thứ tự mẫu giữ nguyên tuyệt đối 0..N-1."""
    loader = DataLoader(sample_dataset, batch_size=3, shuffle=False, one_hot=False)
    all_labels = []
    for _, by in loader:
        all_labels.extend(by.tolist())

    assert all_labels == list(range(10))


@pytest.mark.parametrize("invalid_bs", [0, -4, True, "32"])
def test_dataloader_invalid_batch_size(
    sample_dataset: MnistDataset, invalid_bs: object
) -> None:
    """batch_size không hợp lệ phải ném InvalidConfigError."""
    with pytest.raises(InvalidConfigError):
        DataLoader(sample_dataset, batch_size=invalid_bs)  # type: ignore[arg-type]


def test_dataloader_empty_dataset() -> None:
    """DataLoader xử lý trơn tru tập dữ liệu rỗng."""
    empty_ds = MnistDataset(
        np.zeros((0, 784), dtype=np.float32), np.zeros(0, dtype=np.int64)
    )
    loader = DataLoader(empty_ds, batch_size=4)
    assert len(loader) == 0
    batches = list(loader)
    assert len(batches) == 0


def test_one_hot_encoder_and_normalizer() -> None:
    """Kiểm thử trực tiếp các lớp Transforms."""
    encoder = OneHotEncoder(num_classes=5)
    labels = np.array([0, 2, 4], dtype=np.int64)
    oh = encoder(labels)
    assert oh.shape == (3, 5)
    assert oh[0, 0] == 1.0
    assert oh[1, 2] == 1.0
    assert oh[2, 4] == 1.0

    # Lỗi nhãn vượt quá số lớp
    with pytest.raises(InvalidConfigError):
        encoder(np.array([5]))
    with pytest.raises(InvalidConfigError):
        encoder(np.array([-1]))
    with pytest.raises(InvalidConfigError):
        OneHotEncoder(num_classes=0)

    # Normalizer
    norm = Normalizer(scale=255.0)
    arr = np.array([0.0, 127.5, 255.0], dtype=np.float32)
    res = norm(arr)
    assert np.allclose(res, [0.0, 0.5, 1.0])

    with pytest.raises(InvalidConfigError):
        Normalizer(scale=-10.0)
