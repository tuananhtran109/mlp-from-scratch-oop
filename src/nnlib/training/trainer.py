"""Bộ điều phối vòng lặp huấn luyện và đánh giá mạng nơ-ron (Trainer)."""

from __future__ import annotations

import time
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

from nnlib.exceptions import InvalidConfigError, ShapeMismatchError
from nnlib.layers.base import Layer
from nnlib.optim.base import IOptimizer
from nnlib.tensor import Tensor
from nnlib.training.callbacks import ICallback
from nnlib.training.history import TrainingHistory


class Trainer:
    """Điều phối huấn luyện (Vai trò: Context trong Strategy, Subject trong Observer).

    Thuộc tính:
        _model (Layer): Mô hình mạng nơ-ron cần huấn luyện (Aggregation).
        _loss (Any): Hàm mất mát cài đặt ``forward`` và ``backward`` (Strategy).
        _optimizer (IOptimizer): Thuật toán cập nhật trọng số (Strategy).
        _callbacks (list[ICallback]): Danh sách observer nhận sự kiện huấn luyện.
        _history (TrainingHistory): Lịch sử chỉ số theo epoch (Composition).
        _stop_requested (bool): Cờ yêu cầu dừng sớm vòng lặp huấn luyện.
    """

    def __init__(
        self,
        model: Layer,
        loss_fn: Any,
        optimizer: IOptimizer,
        callbacks: Sequence[ICallback] | None = None,
    ) -> None:
        """Khởi tạo Trainer với mô hình, hàm mất mát, optimizer và callbacks."""
        if not isinstance(model, Layer):
            raise InvalidConfigError(
                f"model phải là thực thể của Layer, nhận được: {model!r}."
            )
        if (
            isinstance(loss_fn, type)
            or not callable(getattr(loss_fn, "forward", None))
            or not callable(getattr(loss_fn, "backward", None))
        ):
            raise InvalidConfigError(
                "loss_fn phải là đối tượng thực thi giao diện ILossFunction "
                "(có phương thức forward và backward)."
            )
        if not isinstance(optimizer, IOptimizer):
            raise InvalidConfigError(
                f"optimizer phải thực thi IOptimizer, nhận được: {optimizer!r}."
            )

        self._model: Layer = model
        self._loss: Any = loss_fn
        self._optimizer: IOptimizer = optimizer
        self._history: TrainingHistory = TrainingHistory()
        self._callbacks: list[ICallback] = []
        self._stop_requested: bool = False

        if callbacks is not None:
            for cb in callbacks:
                self.add_callback(cb)

    @property
    def model(self) -> Layer:
        """Trả về mô hình đang được huấn luyện."""
        return self._model

    @property
    def loss(self) -> Any:
        """Trả về hàm mất mát đang sử dụng."""
        return self._loss

    @property
    def optimizer(self) -> IOptimizer:
        """Trả về thuật toán tối ưu hóa đang sử dụng."""
        return self._optimizer

    @property
    def callbacks(self) -> list[ICallback]:
        """Trả về bản sao danh sách các callback đã đăng ký."""
        return list(self._callbacks)

    @property
    def history(self) -> TrainingHistory:
        """Trả về đối tượng lưu lịch sử huấn luyện của Trainer."""
        return self._history

    @property
    def stop_requested(self) -> bool:
        """Trả về True nếu có yêu cầu dừng sớm vòng lặp huấn luyện."""
        return self._stop_requested

    def add_callback(self, callback: ICallback) -> None:
        """Đăng ký thêm một callback và gắn tham chiếu Trainer cho callback đó."""
        if not isinstance(callback, ICallback):
            raise InvalidConfigError(
                f"callback phải kế thừa ICallback, nhận được: {callback!r}."
            )
        callback.set_trainer(self)
        self._callbacks.append(callback)

    def request_stop(self) -> None:
        """Đặt cờ dừng sớm vòng lặp huấn luyện sau khi kết thúc epoch hiện tại."""
        self._stop_requested = True

    @staticmethod
    def _wrap_tensor(data: np.ndarray | Tensor) -> Tensor:
        """Bọc mảng NumPy thành Tensor tại tầng huấn luyện."""
        if isinstance(data, Tensor):
            return data
        return Tensor(data)

    @staticmethod
    def _count_correct(y_pred: Tensor, y_true: Tensor) -> int:
        """Đếm số mẫu dự đoán đúng cho bài toán đa lớp và nhị phân."""
        if y_pred.ndim == 2 and y_pred.shape[1] > 1:
            pred_labels = np.argmax(y_pred.data, axis=1)
            if y_true.ndim == 2 and y_true.shape == y_pred.shape:
                true_labels = np.argmax(y_true.data, axis=1)
            elif y_true.ndim == 1 and y_true.shape[0] == y_pred.shape[0]:
                true_labels = np.asarray(y_true.data, dtype=np.int64)
            else:
                raise ShapeMismatchError(
                    f"Nhãn y_true shape {y_true.shape} không khớp dự đoán "
                    f"shape {y_pred.shape}."
                )
            return int(np.sum(pred_labels == true_labels))

        if y_pred.shape != y_true.shape and y_pred.data.size != y_true.data.size:
            raise ShapeMismatchError(
                f"Nhãn y_true shape {y_true.shape} không khớp dự đoán "
                f"shape {y_pred.shape}."
            )
        pred_binary = (y_pred.data.reshape(-1) >= 0.5).astype(np.int64)
        true_binary = (y_true.data.reshape(-1) >= 0.5).astype(np.int64)
        return int(np.sum(pred_binary == true_binary))

    def _run_train_epoch(
        self,
        train_loader: Iterable[tuple[np.ndarray, np.ndarray]],
    ) -> tuple[float, float]:
        """Thực thi một epoch huấn luyện trên toàn bộ các mini-batch."""
        self._model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for x_batch, y_batch in train_loader:
            x_tensor = self._wrap_tensor(x_batch)
            y_tensor = self._wrap_tensor(y_batch)
            if x_tensor.ndim == 0 or x_tensor.shape[0] <= 0:
                raise ShapeMismatchError(
                    f"Batch đầu vào phải có kích thước >= 1, nhận: {x_tensor.shape}."
                )

            batch_size = x_tensor.shape[0]
            params = self._model.parameters()
            self._optimizer.zero_grad(params)

            y_pred = self._model.forward(x_tensor)
            loss_val = float(self._loss.forward(y_pred, y_tensor))
            grad_loss = self._loss.backward()
            self._model.backward(grad_loss)
            self._optimizer.step(params)

            total_loss += loss_val * batch_size
            total_correct += self._count_correct(y_pred, y_tensor)
            total_samples += batch_size

        if total_samples == 0:
            raise InvalidConfigError("train_loader không chứa mẫu dữ liệu nào.")

        return total_loss / total_samples, total_correct / total_samples

    def evaluate(
        self,
        data_loader: Iterable[tuple[np.ndarray, np.ndarray]],
    ) -> dict[str, float]:
        """Đánh giá mô hình ở chế độ eval trên tập dữ liệu cho trước."""
        self._model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for x_batch, y_batch in data_loader:
            x_tensor = self._wrap_tensor(x_batch)
            y_tensor = self._wrap_tensor(y_batch)
            if x_tensor.ndim == 0 or x_tensor.shape[0] <= 0:
                raise ShapeMismatchError(
                    f"Batch đánh giá phải có kích thước >= 1, nhận: {x_tensor.shape}."
                )

            batch_size = x_tensor.shape[0]
            y_pred = self._model.forward(x_tensor)
            loss_val = float(self._loss.forward(y_pred, y_tensor))

            total_loss += loss_val * batch_size
            total_correct += self._count_correct(y_pred, y_tensor)
            total_samples += batch_size

        if total_samples == 0:
            raise InvalidConfigError("data_loader đánh giá không chứa mẫu dữ liệu nào.")

        return {
            "loss": total_loss / total_samples,
            "accuracy": total_correct / total_samples,
        }

    def fit(
        self,
        train_loader: Iterable[tuple[np.ndarray, np.ndarray]],
        epochs: int,
        val_loader: Iterable[tuple[np.ndarray, np.ndarray]] | None = None,
    ) -> TrainingHistory:
        """Huấn luyện mô hình qua số epoch chỉ định và phát sự kiện tới callback."""
        if (
            isinstance(epochs, (bool, np.bool_))
            or not isinstance(epochs, (int, np.integer))
            or int(epochs) <= 0
        ):
            raise InvalidConfigError(
                f"Số epoch phải là số nguyên dương, nhận được: {epochs!r}."
            )

        num_epochs = int(epochs)
        self._stop_requested = False
        self._history.clear()

        begin_logs: dict[str, Any] = {"epochs": num_epochs}
        for cb in self._callbacks:
            cb.on_train_begin(dict(begin_logs))

        for epoch in range(1, num_epochs + 1):
            epoch_start = time.perf_counter()
            train_loss, train_acc = self._run_train_epoch(train_loader)

            epoch_logs: dict[str, float] = {
                "train_loss": train_loss,
                "train_acc": train_acc,
            }
            if val_loader is not None:
                val_metrics = self.evaluate(val_loader)
                epoch_logs["val_loss"] = val_metrics["loss"]
                epoch_logs["val_acc"] = val_metrics["accuracy"]

            epoch_logs["elapsed_s"] = time.perf_counter() - epoch_start
            self._history.record(epoch_logs)

            for cb in self._callbacks:
                cb.on_epoch_end(epoch, dict(epoch_logs))

            if self._stop_requested:
                break

        end_logs: dict[str, Any] = dict(self._history.latest() or {})
        end_logs["stopped_early"] = self._stop_requested
        for cb in self._callbacks:
            cb.on_train_end(dict(end_logs))

        return self._history
