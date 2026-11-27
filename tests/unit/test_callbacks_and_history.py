"""Kiểm thử đơn vị cho TrainingHistory, ICallback và LoggingCallback."""

from __future__ import annotations

import logging
from abc import ABC

import numpy as np
import pytest

from nnlib.exceptions import InvalidConfigError, StateError
from nnlib.training import ICallback, LoggingCallback, TrainingHistory


class _OnlyEpochCallback(ICallback):
    """Callback thử nghiệm chỉ ghi đè on_epoch_end để kiểm tra interface rỗng."""

    def __init__(self) -> None:
        super().__init__()
        self.epochs_seen: list[int] = []

    def on_epoch_end(self, epoch: int, logs: dict[str, object] | None = None) -> None:
        """Ghi lại chỉ số epoch đã chạy qua."""
        self.epochs_seen.append(epoch)


def test_training_history_record_and_queries() -> None:
    """TrainingHistory lưu đúng các epoch và truy xuất theo metric chính xác."""
    history = TrainingHistory()
    assert len(history) == 0
    assert history.latest() is None
    assert history.to_dict() == {}

    row1 = {"train_loss": 0.9, "train_acc": 0.55}
    row2 = {"train_loss": 0.4, "train_acc": 0.85}
    history.record(row1)
    history.record(row2)

    assert len(history) == 2
    assert history.get_metric("train_loss") == pytest.approx([0.9, 0.4])
    assert history.get_metric("train_acc") == pytest.approx([0.55, 0.85])
    assert history.to_dict() == {
        "train_loss": pytest.approx([0.9, 0.4]),
        "train_acc": pytest.approx([0.55, 0.85]),
    }
    assert history.latest() == {"train_loss": 0.4, "train_acc": 0.85}


def test_training_history_encapsulation_and_clear() -> None:
    """Sửa dữ liệu trả về hoặc dict đầu vào không làm hỏng trạng thái nội bộ."""
    history = TrainingHistory()
    original = {"train_loss": 0.5}
    history.record(original)
    original["train_loss"] = 999.0

    records_copy = history.records
    records_copy[0]["train_loss"] = -1.0
    records_copy.append({"train_loss": 123.0})

    latest_copy = history.latest()
    assert latest_copy is not None
    latest_copy["train_loss"] = 888.0

    assert len(history) == 1
    assert history.get_metric("train_loss") == pytest.approx([0.5])

    history.clear()
    assert len(history) == 0
    assert history.latest() is None


def test_training_history_invalid_inputs_and_state() -> None:
    """Đầu vào sai kiểu, rỗng hoặc khóa thiếu phải ném lỗi thuộc NNError."""
    history = TrainingHistory()

    with pytest.raises(StateError):
        history.get_metric("train_loss")

    with pytest.raises(InvalidConfigError):
        history.record({})

    with pytest.raises(InvalidConfigError):
        history.record({"": 1.0})

    with pytest.raises(InvalidConfigError):
        history.record({"train_loss": True})  # type: ignore[dict-item]

    with pytest.raises(InvalidConfigError):
        history.record({"train_loss": np.bool_(False)})  # type: ignore[dict-item]

    with pytest.raises(InvalidConfigError):
        history.record({"train_loss": "0.5"})  # type: ignore[dict-item]

    history.record({"train_loss": 0.3})
    with pytest.raises(InvalidConfigError):
        history.get_metric("val_loss")

    with pytest.raises(InvalidConfigError):
        history.get_metric("  ")


def test_icallback_contract_and_default_hooks() -> None:
    """ICallback kế thừa ABC và cung cấp các hook mặc định rỗng."""
    assert issubclass(ICallback, ABC)
    cb = _OnlyEpochCallback()
    assert cb.trainer is None

    fake_trainer = object()
    cb.set_trainer(fake_trainer)  # type: ignore[arg-type]
    assert cb.trainer is fake_trainer

    # Gọi hook mặc định không gây lỗi
    cb.on_train_begin({"epochs": 2})
    cb.on_epoch_end(1, {"train_loss": 0.5})
    cb.on_epoch_end(2, {"train_loss": 0.2})
    cb.on_train_end({"train_loss": 0.2})

    assert cb.epochs_seen == [1, 2]


def test_logging_callback_writes_to_logger_without_print(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """LoggingCallback ghi nhật ký qua logging và không dùng print."""
    logger = logging.getLogger("tests.training.logging_cb")
    cb = LoggingCallback(logger=logger, level=logging.INFO)
    assert cb.logger is logger
    assert cb.level == logging.INFO

    with caplog.at_level(logging.INFO, logger=logger.name):
        cb.on_train_begin({"epochs": 3})
        cb.on_epoch_end(1, {"train_loss": 0.45678, "train_acc": 0.8125})
        cb.on_train_end({"stopped_early": False})

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""

    messages = [rec.message for rec in caplog.records]
    assert len(messages) == 3
    assert "Bắt đầu huấn luyện" in messages[0]
    assert "Epoch 1:" in messages[1]
    assert "train_loss=0.4568" in messages[1]
    assert "train_acc=0.8125" in messages[1]
    assert "Kết thúc huấn luyện" in messages[2]


@pytest.mark.parametrize(
    ("bad_logger", "bad_level"),
    [
        ("not_a_logger", logging.INFO),
        (None, True),
        (None, -10),
        (None, "INFO"),
    ],
)
def test_logging_callback_invalid_config(
    bad_logger: object,
    bad_level: object,
) -> None:
    """Cấu hình logger hoặc level sai kiểu/âm phải ném InvalidConfigError."""
    with pytest.raises(InvalidConfigError):
        LoggingCallback(
            logger=bad_logger,  # type: ignore[arg-type]
            level=bad_level,  # type: ignore[arg-type]
        )
