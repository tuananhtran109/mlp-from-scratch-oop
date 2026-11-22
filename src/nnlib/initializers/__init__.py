"""Package chứa interface và các chiến lược khởi tạo trọng số."""

from nnlib.initializers.base import IInitializer
from nnlib.initializers.initializers import (
    HeInitializer,
    XavierInitializer,
    ZerosInitializer,
)

__all__ = [
    "HeInitializer",
    "IInitializer",
    "XavierInitializer",
    "ZerosInitializer",
]
