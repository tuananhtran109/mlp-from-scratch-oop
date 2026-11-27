"""Kiểm thử tích hợp Trainer học bài toán XOR với Layer thật và DataLoader."""

from __future__ import annotations

import numpy as np

from digitapp.data.dataloader import DataLoader
from digitapp.data.dataset import IDataset
from nnlib.initializers import HeInitializer, XavierInitializer
from nnlib.layers.activations import Sigmoid, Tanh
from nnlib.layers.dense import Dense
from nnlib.optim.adam import Adam
from nnlib.optim.sgd import SGD
from nnlib.tensor import Tensor
from nnlib.training import Trainer
from tests.fakes.fake_loss import FakeMSELoss
from tests.fakes.fake_sequential import FakeSequential


class _XorDataset(IDataset):
    """Tập dữ liệu 4 mẫu cho bài toán cổng logic XOR."""

    def __init__(self) -> None:
        self._x = np.array(
            [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
            dtype=np.float32,
        )
        self._y = np.array([[0.0], [1.0], [1.0], [0.0]], dtype=np.float32)

    def __len__(self) -> int:
        return len(self._x)

    def __getitem__(
        self, idx: int | slice | np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        return self._x[idx], self._y[idx]


def _build_xor_network() -> FakeSequential:
    """Dựng mạng 2 -> 8 -> 1 từ các lớp Dense, Tanh, Sigmoid thật của Người A."""
    return FakeSequential(
        [
            Dense(2, 8, weight_initializer=HeInitializer(seed=42)),
            Tanh(),
            Dense(8, 1, weight_initializer=XavierInitializer(seed=43)),
            Sigmoid(),
        ]
    )


def test_trainer_learns_xor_with_adam_t15() -> None:
    """Kiểm thử T15: Trainer + Adam làm giảm loss rõ rệt và giải đúng XOR."""
    dataset = _XorDataset()
    loader = DataLoader(dataset, batch_size=4, shuffle=False, one_hot=False)
    model = _build_xor_network()
    trainer = Trainer(
        model=model,
        loss_fn=FakeMSELoss(),
        optimizer=Adam(lr=0.05),
    )

    history = trainer.fit(loader, epochs=250)
    losses = history.get_metric("train_loss")

    assert losses[-1] < losses[0] * 0.1
    assert losses[-1] < 0.02

    metrics = trainer.evaluate(loader)
    assert metrics["accuracy"] == 1.0

    preds = model.forward(Tensor(dataset._x)).data
    assert preds[0, 0] < 0.2
    assert preds[1, 0] > 0.8
    assert preds[2, 0] > 0.8
    assert preds[3, 0] < 0.2


def test_trainer_learns_xor_with_sgd_momentum_and_dataloader() -> None:
    """Kiểm chứng pipeline DataLoader(shuffle=True) -> Trainer -> SGD trên XOR."""
    dataset = _XorDataset()
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        seed=42,
        one_hot=False,
    )
    model = _build_xor_network()
    trainer = Trainer(
        model=model,
        loss_fn=FakeMSELoss(),
        optimizer=SGD(lr=0.3, momentum=0.9),
    )

    history = trainer.fit(loader, epochs=350)
    losses = history.get_metric("train_loss")

    assert losses[-1] < losses[0] * 0.1
    assert losses[-1] < 0.02

    eval_loader = DataLoader(dataset, batch_size=4, shuffle=False, one_hot=False)
    metrics = trainer.evaluate(eval_loader)
    assert metrics["accuracy"] == 1.0
