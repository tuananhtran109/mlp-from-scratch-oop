"""Kiểm thử đơn vị cho resource_path và AppConfig (Chỉ tiêu T24)."""

import json
import sys
from pathlib import Path

import pytest

from digitapp.common.paths import get_project_root, resource_path
from digitapp.config.app_config import AppConfig
from nnlib.exceptions import InvalidConfigError, StateError


def test_resource_path_and_project_root() -> None:
    """resource_path giải quyết đường dẫn chính xác và độc lập với cwd."""
    root = get_project_root()
    assert root.is_dir()
    assert (root / "pyproject.toml").is_file()

    cfg_path = resource_path("config/config.json")
    assert cfg_path.is_absolute()
    assert cfg_path.is_file()

    # Kiểm thử đường dẫn tuyệt đối truyền vào
    assert resource_path(cfg_path) == cfg_path


def test_resource_path_frozen_simulation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mô phỏng môi trường PyInstaller frozen với sys._MEIPASS."""
    fake_meipass = Path("/fake/bundled/app")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(fake_meipass), raising=False)

    resolved = resource_path("config/config.json")
    assert str(resolved).startswith(str(fake_meipass.resolve()))


def test_app_config_load_and_properties_t24() -> None:
    """Kiểm thử T24: Nạp tệp cấu hình mặc định và kiểm tra toàn bộ thuộc tính."""
    config = AppConfig()
    config.load()

    assert config.raw_dir == "data/raw"
    assert config.val_ratio == 0.1
    assert config.seed == 42
    assert config.epochs == 15
    assert config.batch_size == 128
    assert config.loss_name == "cross_entropy"
    assert config.optimizer_name == "adam"
    assert config.learning_rate == 0.001
    assert config.model_path == "models/mnist_mlp"
    assert config.confidence_threshold == 0.6
    assert len(config.model_layers) == 7

    assert isinstance(config.data_config, dict)
    assert isinstance(config.model_config, dict)
    assert isinstance(config.training_config, dict)
    assert isinstance(config.app_config, dict)
    assert isinstance(config.values, dict)


def test_app_config_state_error_before_load() -> None:
    """Truy cập thuộc tính hoặc gọi validate trước khi load() phải ném StateError."""
    config = AppConfig()

    with pytest.raises(StateError):
        _ = config.batch_size

    with pytest.raises(StateError):
        _ = config.values

    with pytest.raises(StateError):
        config.validate()


def test_app_config_missing_file_t24() -> None:
    """Kiểm thử T24: Tệp cấu hình không tồn tại ném InvalidConfigError."""
    config = AppConfig("config/non_existent_file.json")
    with pytest.raises(InvalidConfigError) as exc_info:
        config.load()
    assert "Không tìm thấy tệp cấu hình" in str(exc_info.value)


def test_app_config_invalid_json(tmp_path: Path) -> None:
    """Tệp cấu hình sai định dạng JSON ném InvalidConfigError."""
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{ unquoted_key: 123, }", encoding="utf-8")

    config = AppConfig(bad_json)
    with pytest.raises(InvalidConfigError) as exc_info:
        config.load()
    assert "không đúng định dạng JSON" in str(exc_info.value)


@pytest.mark.parametrize("missing_key", ["data", "model", "training", "app"])
def test_app_config_missing_root_keys_t24(tmp_path: Path, missing_key: str) -> None:
    """Kiểm thử T24: Thiếu bất kỳ khóa gốc nào phải ném InvalidConfigError."""
    valid_cfg = {
        "data": {"raw_dir": "data/raw", "val_ratio": 0.1, "seed": 42},
        "model": {"layers": [{"type": "dense"}]},
        "training": {
            "epochs": 10,
            "batch_size": 32,
            "loss": "cross_entropy",
            "optimizer": {"name": "sgd", "lr": 0.1},
        },
        "app": {"model_path": "models/m", "confidence_threshold": 0.5},
    }
    del valid_cfg[missing_key]

    cfg_file = tmp_path / f"missing_{missing_key}.json"
    cfg_file.write_text(json.dumps(valid_cfg), encoding="utf-8")

    config = AppConfig(cfg_file)
    with pytest.raises(InvalidConfigError) as exc_info:
        config.load()
    assert missing_key in str(exc_info.value)


@pytest.mark.parametrize(
    ("section", "key", "invalid_value"),
    [
        ("data", "val_ratio", 0.0),
        ("data", "val_ratio", 1.0),
        ("data", "val_ratio", -0.2),
        ("data", "val_ratio", True),  # loại trừ bool
        ("data", "seed", True),  # loại trừ bool
        ("data", "raw_dir", ""),
        ("training", "epochs", 0),
        ("training", "epochs", -5),
        ("training", "epochs", True),  # loại trừ bool
        ("training", "batch_size", 0),
        ("training", "batch_size", True),  # loại trừ bool
        ("training", "loss", "   "),
        ("app", "model_path", ""),
        ("app", "confidence_threshold", -0.01),
        ("app", "confidence_threshold", 1.01),
        ("app", "confidence_threshold", True),  # loại trừ bool
        ("model", "layers", []),
    ],
)
def test_app_config_invalid_field_values(
    tmp_path: Path, section: str, key: str, invalid_value: object
) -> None:
    """Các giá trị cấu hình vi phạm điều kiện biên ném InvalidConfigError."""
    valid_cfg = {
        "data": {"raw_dir": "data/raw", "val_ratio": 0.1, "seed": 42},
        "model": {"layers": [{"type": "dense"}]},
        "training": {
            "epochs": 10,
            "batch_size": 32,
            "loss": "cross_entropy",
            "optimizer": {"name": "sgd", "lr": 0.1},
        },
        "app": {"model_path": "models/m", "confidence_threshold": 0.5},
    }
    valid_cfg[section][key] = invalid_value

    cfg_file = tmp_path / "invalid_val.json"
    cfg_file.write_text(json.dumps(valid_cfg), encoding="utf-8")

    config = AppConfig(cfg_file)
    with pytest.raises(InvalidConfigError):
        config.load()


@pytest.mark.parametrize(
    "invalid_lr",
    [0.0, -0.01, True, float("nan"), float("inf")],
)
def test_app_config_invalid_optimizer_lr(tmp_path: Path, invalid_lr: object) -> None:
    """Kiểm tra điều kiện biên cho learning rate trong cấu hình."""
    cfg = {
        "data": {"raw_dir": "data/raw", "val_ratio": 0.1, "seed": 42},
        "model": {"layers": [{"type": "dense"}]},
        "training": {
            "epochs": 10,
            "batch_size": 32,
            "loss": "cross_entropy",
            "optimizer": {"name": "sgd", "lr": invalid_lr},
        },
        "app": {"model_path": "models/m", "confidence_threshold": 0.5},
    }
    cfg_file = tmp_path / "invalid_lr.json"
    cfg_file.write_text(json.dumps(cfg), encoding="utf-8")

    config = AppConfig(cfg_file)
    with pytest.raises(InvalidConfigError):
        config.load()
