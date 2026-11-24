"""Package thuật toán tối ưu hóa cho mạng nơ-ron nnlib."""

from nnlib.optim.base import IOptimizer
from nnlib.optim.sgd import SGD

__all__ = ["IOptimizer", "SGD"]
