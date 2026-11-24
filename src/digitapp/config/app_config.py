"""Quản lý, nạp và kiểm tra tính toàn vẹn của tệp cấu hình ứng dụng."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from digitapp.common.paths import resource_path
from nnlib.exceptions import InvalidConfigError, StateError


class AppConfig:
    """Lớp quản lý cấu hình hệ thống từ tệp JSON."""

    def __init__(self, path: str | Path | None = None) -> None:
        """Khởi tạo AppConfig với đường dẫn tệp (mặc định config/config.json)."""
        self._path: Path = resource_path(path or "config/config.json")
        self._values: dict[str, Any] | None = None

    @property
    def path(self) -> Path:
        """Trả về đường dẫn tuyệt đối tới tệp cấu hình."""
        return self._path

    def load(self) -> None:
        """Đọc tệp JSON từ đường dẫn và kích hoạt kiểm tra tính hợp lệ."""
        if not self._path.is_file():
            raise InvalidConfigError(f"Không tìm thấy tệp cấu hình tại: {self._path}")

        try:
            with open(self._path, "r", encoding="utf-8") as f:
                self._values = json.load(f)
        except json.JSONDecodeError as exc:
            raise InvalidConfigError(
                f"Tệp cấu hình không đúng định dạng JSON: {exc}"
            ) from exc
        except OSError as exc:
            raise InvalidConfigError(f"Lỗi khi đọc tệp cấu hình: {exc}") from exc

        self.validate()

    def validate(self) -> None:
        """Kiểm tra sự tồn tại của các nhóm khóa bắt buộc và kiểu giá trị."""
        if self._values is None:
            raise StateError(
                "Chưa nạp cấu hình. Vui lòng gọi load() trước khi validate()."
            )

        # Kiểm tra 4 nhóm khóa gốc
        required_roots = ("data", "model", "training", "app")
        for key in required_roots:
            if key not in self._values or not isinstance(self._values[key], dict):
                raise InvalidConfigError(f"Cấu hình thiếu khóa gốc bắt buộc: '{key}'.")

        # 1. Kiểm tra nhóm "data"
        data = self._values["data"]
        raw_dir = data.get("raw_dir")
        if not isinstance(raw_dir, str) or not raw_dir.strip():
            raise InvalidConfigError(
                f"Trường 'data.raw_dir' phải là chuỗi không rỗng, "
                f"nhận được: {raw_dir!r}."
            )

        val_ratio = data.get("val_ratio")
        if (
            isinstance(val_ratio, bool)
            or not isinstance(val_ratio, (int, float))
            or np.isnan(val_ratio)
            or not (0.0 < float(val_ratio) < 1.0)
        ):
            raise InvalidConfigError(
                f"Trường 'data.val_ratio' phải là số thực trong khoảng (0, 1), "
                f"nhận được: {val_ratio!r}."
            )

        seed = data.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
            raise InvalidConfigError(
                f"Trường 'data.seed' phải là số nguyên, nhận được: {seed!r}."
            )

        # 2. Kiểm tra nhóm "training"
        training = self._values["training"]
        epochs = training.get("epochs")
        if (
            isinstance(epochs, bool)
            or not isinstance(epochs, (int, np.integer))
            or int(epochs) <= 0
        ):
            raise InvalidConfigError(
                f"Trường 'training.epochs' phải là số nguyên dương, "
                f"nhận được: {epochs!r}."
            )

        batch_size = training.get("batch_size")
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, (int, np.integer))
            or int(batch_size) <= 0
        ):
            raise InvalidConfigError(
                f"Trường 'training.batch_size' phải là số nguyên dương, "
                f"nhận được: {batch_size!r}."
            )

        loss = training.get("loss")
        if not isinstance(loss, str) or not loss.strip():
            raise InvalidConfigError(
                f"Trường 'training.loss' phải là chuỗi không rỗng, nhận được: {loss!r}."
            )

        opt = training.get("optimizer")
        if not isinstance(opt, dict):
            raise InvalidConfigError(
                f"Trường 'training.optimizer' phải là một dictionary, "
                f"nhận được: {opt!r}."
            )

        opt_name = opt.get("name")
        if not isinstance(opt_name, str) or not opt_name.strip():
            raise InvalidConfigError(
                f"Trường 'training.optimizer.name' phải là chuỗi không rỗng, "
                f"nhận được: {opt_name!r}."
            )

        lr = opt.get("lr")
        if (
            isinstance(lr, bool)
            or not isinstance(lr, (int, float))
            or np.isnan(lr)
            or np.isinf(lr)
            or float(lr) <= 0.0
        ):
            raise InvalidConfigError(
                f"Trường 'training.optimizer.lr' phải là số thực dương, "
                f"nhận được: {lr!r}."
            )

        # 3. Kiểm tra nhóm "app"
        app = self._values["app"]
        model_path = app.get("model_path")
        if not isinstance(model_path, str) or not model_path.strip():
            raise InvalidConfigError(
                f"Trường 'app.model_path' phải là chuỗi không rỗng, "
                f"nhận được: {model_path!r}."
            )

        conf = app.get("confidence_threshold")
        if (
            isinstance(conf, bool)
            or not isinstance(conf, (int, float))
            or np.isnan(conf)
            or not (0.0 <= float(conf) <= 1.0)
        ):
            raise InvalidConfigError(
                f"Trường 'app.confidence_threshold' phải là số thực trong đoạn [0, 1], "
                f"nhận được: {conf!r}."
            )

        # 4. Kiểm tra nhóm "model"
        model = self._values["model"]
        layers = model.get("layers")
        if not isinstance(layers, list) or len(layers) == 0:
            raise InvalidConfigError(
                "Trường 'model.layers' phải là một danh sách các layer không rỗng."
            )

    def _ensure_loaded(self) -> dict[str, Any]:
        """Đảm bảo cấu hình đã được nạp trước khi truy cập thuộc tính."""
        if self._values is None:
            raise StateError(
                "Chưa nạp cấu hình. Vui lòng gọi load() trước khi truy cập thuộc tính."
            )
        return self._values

    @property
    def values(self) -> dict[str, Any]:
        """Trả về toàn bộ từ điển cấu hình."""
        return self._ensure_loaded()

    @property
    def data_config(self) -> dict[str, Any]:
        """Trả về cấu hình phân vùng dữ liệu."""
        return self._ensure_loaded()["data"]

    @property
    def model_config(self) -> dict[str, Any]:
        """Trả về cấu hình mô hình."""
        return self._ensure_loaded()["model"]

    @property
    def training_config(self) -> dict[str, Any]:
        """Trả về cấu hình huấn luyện."""
        return self._ensure_loaded()["training"]

    @property
    def app_config(self) -> dict[str, Any]:
        """Trả về cấu hình ứng dụng."""
        return self._ensure_loaded()["app"]

    @property
    def raw_dir(self) -> str:
        """Trả về đường dẫn tương đối tới thư mục dữ liệu thô."""
        return str(self.data_config["raw_dir"])

    @property
    def val_ratio(self) -> float:
        """Trả về tỉ lệ dữ liệu dành cho validation."""
        return float(self.data_config["val_ratio"])

    @property
    def seed(self) -> int:
        """Trả về giá trị seed ngẫu nhiên dùng chung."""
        return int(self.data_config["seed"])

    @property
    def epochs(self) -> int:
        """Trả về số lượng epoch huấn luyện."""
        return int(self.training_config["epochs"])

    @property
    def batch_size(self) -> int:
        """Trả về kích thước mini-batch."""
        return int(self.training_config["batch_size"])

    @property
    def loss_name(self) -> str:
        """Trả về tên hàm mất mát."""
        return str(self.training_config["loss"])

    @property
    def optimizer_name(self) -> str:
        """Trả về tên thuật toán tối ưu hóa."""
        return str(self.training_config["optimizer"]["name"])

    @property
    def learning_rate(self) -> float:
        """Trả về tốc độ học (learning rate)."""
        return float(self.training_config["optimizer"]["lr"])

    @property
    def model_path(self) -> str:
        """Trả về đường dẫn lưu/nạp mô hình."""
        return str(self.app_config["model_path"])

    @property
    def confidence_threshold(self) -> float:
        """Trả về ngưỡng xác suất nhận diện tin cậy."""
        return float(self.app_config["confidence_threshold"])

    @property
    def model_layers(self) -> list[dict[str, Any]]:
        """Trả về danh sách cấu hình từng layer trong mô hình."""
        return list(self.model_config["layers"])
