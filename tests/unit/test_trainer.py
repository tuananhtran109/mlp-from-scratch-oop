"""Kiểm thử đơn vị cho Trainer (chỉ tiêu callback và điều phối huấn luyện)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from nnlib.exceptions import InvalidConfigError
from nnlib.initializers import HeInitializer, XavierInitializer
from nnlib.layers.base import Layer
from nnlib.layers.dense import Dense
from nnlib.layers.dropout import Dropout
from nnlib.optim.adam import Adam
from nnlib.optim.sgd import SGD
from nnlib.tensor import Parameter, Tensor
from nnlib.training import ICallback, Trainer
from tests.fakes.fake_loss import FakeMSELoss
from tests.fakes.fake_sequential import FakeSequential


class _RecordingCallback(ICallback):
    """Callback ghi nhận toàn bộ thứ tự sự kiện kèm tên định danh."""

    def __init__(
        self,
        tag: str,
        sink: list[tuple[str, str, int | None]],
        stop_at_epoch: int | None = None,
    ) -> None:
        super().__init__()
        self.tag = tag
        self.sink = sink
        self.stop_at_epoch = stop_at_epoch
        self.last_logs: dict[str, Any] | None = None

    def on_train_begin(self, logs: dict[str, Any] | None = None) -> None:
        self.sink.append((self.tag, "begin", None))
        self.last_logs = logs

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        self.sink.append((self.tag, "epoch_end", epoch))
        self.last_logs = logs
        if self.stop_at_epoch is not None and epoch == self.stop_at_epoch:
            assert self.trainer is not None
            self.trainer.request_stop()

    def on_train_end(self, logs: dict[str, Any] | None = None) -> None:
        self.sink.append((self.tag, "end", None))
        self.last_logs = logs


class _ModeSpyLayer(Layer):
    """Layer gián điệp ghi lại trạng thái training=True/False ở mỗi lần forward."""

    def __init__(self) -> None:
        super().__init__(name="ModeSpy")
        self.seen_modes: list[bool] = []
        self.received_tensors: list[bool] = []

    def forward(self, x: Tensor) -> Tensor:
        self.seen_modes.append(self._training)
        self.received_tensors.append(isinstance(x, Tensor))
        return Tensor(x.data.copy())

    def backward(self, grad_out: Tensor) -> Tensor:
        return Tensor(grad_out.data.copy())

    def parameters(self) -> list[Parameter]:
        return []


def test_trainer_callback_order_t16() -> None:
    """Kiểm thử T16: Thứ tự on_train_begin -> on_epoch_end x E -> on_train_end."""
    events: list[tuple[str, str, int | None]] = []
    cb1 = _RecordingCallback("cb1", events)
    cb2 = _RecordingCallback("cb2", events)

    model = Dense(2, 1, weight_initializer=XavierInitializer(seed=1))
    trainer = Trainer(
        model=model,
        loss_fn=FakeMSELoss(),
        optimizer=SGD(lr=0.01, momentum=0.0),
        callbacks=[cb1, cb2],
    )
    assert cb1.trainer is trainer
    assert cb2.trainer is trainer
    assert len(trainer.callbacks) == 2

    train_batches = [
        (
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            np.array([[1.0], [2.0]], dtype=np.float32),
        )
    ]
    val_batches = [
        (
            np.array([[0.5, 1.0]], dtype=np.float32),
            np.array([[0.5]], dtype=np.float32),
        )
    ]

    history = trainer.fit(train_batches, epochs=3, val_loader=val_batches)

    expected_events = [
        ("cb1", "begin", None),
        ("cb2", "begin", None),
        ("cb1", "epoch_end", 1),
        ("cb2", "epoch_end", 1),
        ("cb1", "epoch_end", 2),
        ("cb2", "epoch_end", 2),
        ("cb1", "epoch_end", 3),
        ("cb2", "epoch_end", 3),
        ("cb1", "end", None),
        ("cb2", "end", None),
    ]
    assert events == expected_events
    assert len(history) == 3
    latest = history.latest()
    assert latest is not None
    assert {"train_loss", "train_acc", "val_loss", "val_acc", "elapsed_s"} <= set(
        latest.keys()
    )
    assert cb2.last_logs is not None
    assert cb2.last_logs["stopped_early"] is False


def test_trainer_request_stop_halts_early() -> None:
    """request_stop() dừng vòng lặp đúng sau epoch yêu cầu và vẫn gọi on_train_end."""
    events: list[tuple[str, str, int | None]] = []
    stopper = _RecordingCallback("stopper", events, stop_at_epoch=2)

    model = Dense(2, 1, weight_initializer=HeInitializer(seed=2))
    trainer = Trainer(
        model=model,
        loss_fn=FakeMSELoss(),
        optimizer=Adam(lr=0.01),
    )
    trainer.add_callback(stopper)

    batches = [
        (
            np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            np.array([[1.0], [0.0]], dtype=np.float32),
        )
    ]
    history = trainer.fit(batches, epochs=5)

    assert len(history) == 2
    assert trainer.stop_requested is True
    assert events == [
        ("stopper", "begin", None),
        ("stopper", "epoch_end", 1),
        ("stopper", "epoch_end", 2),
        ("stopper", "end", None),
    ]
    assert stopper.last_logs is not None
    assert stopper.last_logs["stopped_early"] is True


def test_trainer_switches_train_eval_modes_and_wraps_tensor() -> None:
    """Trainer bật train() khi huấn luyện, eval() khi validation và bọc Tensor."""
    spy = _ModeSpyLayer()
    dropout = Dropout(rate=0.5, seed=7)
    dense = Dense(3, 2, weight_initializer=HeInitializer(seed=3))
    model = FakeSequential([spy, dropout, dense])

    trainer = Trainer(
        model=model,
        loss_fn=FakeMSELoss(),
        optimizer=Adam(lr=0.01),
    )

    train_batches = [
        (
            np.array([[1.0, 2.0, 3.0], [0.5, -0.5, 1.0]], dtype=np.float32),
            np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        ),
        (
            np.array([[-1.0, 0.0, 1.0]], dtype=np.float32),
            np.array([[1.0, 0.0]], dtype=np.float32),
        ),
    ]
    val_batches = [
        (
            np.array([[0.2, 0.4, 0.6]], dtype=np.float32),
            np.array([[0.0, 1.0]], dtype=np.float32),
        )
    ]

    trainer.fit(train_batches, epochs=2, val_loader=val_batches)

    # Mỗi epoch: 2 batch train (True, True) + 1 batch val (False)
    assert spy.seen_modes == [True, True, False, True, True, False]
    assert all(spy.received_tensors)

    eval_metrics = trainer.evaluate(val_batches)
    assert spy.seen_modes[-1] is False
    assert model.training is False
    assert dropout.training is False
    assert eval_metrics["loss"] >= 0.0
    assert 0.0 <= eval_metrics["accuracy"] <= 1.0


def test_trainer_weighted_loss_and_multiclass_int_labels() -> None:
    """Loss và accuracy trung bình được tính theo trọng số số mẫu của từng batch."""

    class _ConstantModel(Layer):
        def forward(self, x: Tensor) -> Tensor:
            # Dự đoán cố định lớp 0 cho 3 mẫu đầu, lớp 1 cho mẫu sau
            out = np.zeros((x.shape[0], 2), dtype=np.float32)
            out[:, 0] = 1.0
            return Tensor(out)

        def backward(self, grad_out: Tensor) -> Tensor:
            return Tensor(np.zeros((grad_out.shape[0], 2), dtype=np.float32))

    trainer = Trainer(
        model=_ConstantModel(),
        loss_fn=FakeMSELoss(),
        optimizer=SGD(lr=0.1, momentum=0.0),
    )

    # Batch 1: 3 mẫu đều đúng lớp 0 (loss = 0.0)
    # Batch 2: 1 mẫu sai (true=[0, 1], pred=[1, 0]) -> L_MSE = (1^2 + (-1)^2)/1 = 2.0
    # Trung bình có trọng số = (0.0 * 3 + 2.0 * 1) / 4 = 0.5; accuracy = 3/4 = 0.75
    batches = [
        (
            np.zeros((3, 2), dtype=np.float32),
            np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], dtype=np.float32),
        ),
        (
            np.zeros((1, 2), dtype=np.float32),
            np.array([[0.0, 1.0]], dtype=np.float32),
        ),
    ]
    metrics = trainer.evaluate(batches)
    assert metrics["loss"] == pytest.approx(0.5)
    assert metrics["accuracy"] == pytest.approx(0.75)


def test_trainer_invalid_config_raises_error() -> None:
    """Tham số khởi tạo hoặc tham số fit không hợp lệ ném InvalidConfigError."""
    model = Dense(2, 1)
    loss_fn = FakeMSELoss()
    opt = Adam(lr=0.01)

    with pytest.raises(InvalidConfigError):
        Trainer(model="not_layer", loss_fn=loss_fn, optimizer=opt)  # type: ignore[arg-type]

    with pytest.raises(InvalidConfigError):
        Trainer(model=model, loss_fn=FakeMSELoss, optimizer=opt)

    with pytest.raises(InvalidConfigError):
        Trainer(model=model, loss_fn=loss_fn, optimizer="not_opt")  # type: ignore[arg-type]

    with pytest.raises(InvalidConfigError):
        Trainer(model=model, loss_fn=loss_fn, optimizer=opt, callbacks=["bad"])  # type: ignore[list-item]

    trainer = Trainer(model=model, loss_fn=loss_fn, optimizer=opt)
    with pytest.raises(InvalidConfigError):
        trainer.add_callback("bad")  # type: ignore[arg-type]

    for bad_epochs in (0, -2, True, np.bool_(True), "3"):
        with pytest.raises(InvalidConfigError):
            trainer.fit(
                [
                    (
                        np.zeros((1, 2), dtype=np.float32),
                        np.zeros((1, 1), dtype=np.float32),
                    )
                ],
                epochs=bad_epochs,  # type: ignore[arg-type]
            )

    with pytest.raises(InvalidConfigError):
        trainer.fit([], epochs=1)

    with pytest.raises(InvalidConfigError):
        trainer.evaluate([])
