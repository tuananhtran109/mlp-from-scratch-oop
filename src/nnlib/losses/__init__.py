"""Package chứa interface ILossFunction và các hàm mất mát cụ thể."""

from nnlib.losses.base import ILossFunction
from nnlib.losses.losses import CrossEntropyLoss, MSELoss

__all__ = [
    "CrossEntropyLoss",
    "ILossFunction",
    "MSELoss",
]
