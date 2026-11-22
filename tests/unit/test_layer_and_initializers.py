"""Kiểm thử đơn vị cho Layer (ABC) và các chiến lược khởi tạo trọng số."""

import numpy as np
import pytest

from nnlib.exceptions import InvalidConfigError, ShapeMismatchError, StateError
from nnlib.initializers import (
    HeInitializer,
    IInitializer,
    XavierInitializer,
    ZerosInitializer,
)
from nnlib.layers.base import Layer
from nnlib.tensor import Parameter, Tensor


class _DummyLayer(Layer):
    """Lớp giả hai tham số dùng để kiểm tra các phương thức mặc định của Layer."""

    def __init__(self) -> None:
        super().__init__(name="dummy")
        self._weights = Parameter(np.ones((2, 2), dtype=np.float32), name="weights")
        self._bias = Parameter(np.zeros((2,), dtype=np.float32), name="bias")
        self._cache_x: Tensor | None = None

    def forward(self, x: Tensor) -> Tensor:
        self._cache_x = x
        return x

    def backward(self, grad_out: Tensor) -> Tensor:
        self._require_cache(self._cache_x)
        return grad_out

    def parameters(self) -> list[Parameter]:
        return [self._weights, self._bias]


def test_layer_abc_state_dict_atomic_and_require_cache() -> None:
    """Kiểm tra ABC, train/eval, R2 (_require_cache) và load_state_dict an toàn."""
    with pytest.raises(TypeError):
        Layer()  # type: ignore[abstract]

    layer = _DummyLayer()
    assert layer.name == "dummy"
    assert layer.training is True

    # R2: backward trước forward phải báo StateError
    with pytest.raises(StateError):
        layer.backward(Tensor([[1.0, 2.0]]))
    _ = layer.forward(Tensor([[1.0, 2.0]]))
    assert layer.backward(Tensor([[1.0, 2.0]])).shape == (1, 2)

    layer.eval()
    assert layer.training is False
    layer.train()
    assert layer.training is True

    state = layer.state_dict()
    assert "weights" in state and "bias" in state
    assert np.allclose(state["weights"], 1.0)

    # Sửa bản sao state không làm đổi trọng số trong layer
    state["weights"][0, 0] = 99.0
    assert layer.parameters()[0].data[0, 0] == 1.0

    # Nạp dở dang (weights đúng shape, bias sai shape) không được làm đổi weights
    with pytest.raises(ShapeMismatchError):
        layer.load_state_dict(
            {
                "weights": np.full((2, 2), 9.0, dtype=np.float32),
                "bias": np.zeros((5,), dtype=np.float32),
            }
        )
    assert np.allclose(layer.parameters()[0].data, 1.0)

    # Nạp hợp lệ cập nhật đủ cả hai tham số
    layer.load_state_dict(
        {
            "weights": np.full((2, 2), 3.0, dtype=np.float32),
            "bias": np.full((2,), 0.5, dtype=np.float32),
        }
    )
    assert np.allclose(layer.parameters()[0].data, 3.0)
    assert np.allclose(layer.parameters()[1].data, 0.5)

    with pytest.raises(InvalidConfigError):
        layer.load_state_dict({"weights": np.ones((2, 2), dtype=np.float32)})


def test_he_and_xavier_initializers() -> None:
    """He và Xavier sinh đúng phương sai, hỗ trợ float64 và tái lập theo seed."""
    he1 = HeInitializer(seed=42)
    he2 = HeInitializer(seed=42)
    assert isinstance(he1, IInitializer)
    assert he1.seed == 42

    w_he1 = he1.initialize((1000, 100), dtype=np.float32)
    w_he2 = he2.initialize((1000, 100), dtype=np.float32)
    assert np.array_equal(w_he1, w_he2)
    assert w_he1.shape == (1000, 100)
    assert w_he1.dtype == np.float32
    assert abs(float(w_he1.std()) - np.sqrt(2.0 / 1000)) < 0.005

    xavier = XavierInitializer(seed=7)
    w_xav = xavier.initialize((500, 500), dtype=np.float64)
    assert w_xav.dtype == np.float64
    assert abs(float(w_xav.std()) - np.sqrt(2.0 / 1000)) < 0.005


def test_zeros_initializer_and_invalid_configs() -> None:
    """ZerosInitializer trả mảng 0; shape hoặc seed sai ném InvalidConfigError."""
    zeros = ZerosInitializer()
    b = zeros.initialize((16,), dtype=np.float32)
    assert b.shape == (16,)
    assert b.dtype == np.float32
    assert np.all(b == 0.0)

    with pytest.raises(InvalidConfigError):
        HeInitializer(seed=True)
    with pytest.raises(InvalidConfigError):
        HeInitializer().initialize((10,))
    with pytest.raises(InvalidConfigError):
        XavierInitializer().initialize((10, 0))
    with pytest.raises(InvalidConfigError):
        ZerosInitializer().initialize(())
