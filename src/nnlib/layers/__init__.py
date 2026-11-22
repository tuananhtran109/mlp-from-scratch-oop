"""Package chứa lớp cơ sở Layer và các lớp mạng nơ-ron."""

from nnlib.layers.base import Layer
from nnlib.layers.dense import Dense

__all__ = [
    "Dense",
    "Layer",
]
