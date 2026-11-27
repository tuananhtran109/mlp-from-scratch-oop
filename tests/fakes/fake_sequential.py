"""Lớp FakeSequential giả lập chuỗi Layer khi Sequential chưa có trên develop."""

from __future__ import annotations

from collections.abc import Sequence

from nnlib.exceptions import InvalidConfigError
from nnlib.layers.base import Layer
from nnlib.tensor import Parameter, Tensor


class FakeSequential(Layer):
    """Chuỗi các Layer thật phục vụ kiểm thử nhiều lớp trước khi có Sequential."""

    def __init__(
        self,
        layers: Sequence[Layer],
        name: str | None = None,
    ) -> None:
        """Khởi tạo chuỗi layer từ danh sách các Layer hợp lệ."""
        super().__init__(name=name)
        layer_list = list(layers)
        if not layer_list or not all(isinstance(layer, Layer) for layer in layer_list):
            raise InvalidConfigError(
                "FakeSequential yêu cầu danh sách không rỗng các đối tượng Layer."
            )
        self._layers: list[Layer] = layer_list

    @property
    def layers(self) -> list[Layer]:
        """Trả về bản sao danh sách các lớp con."""
        return list(self._layers)

    def forward(self, x: Tensor) -> Tensor:
        """Lan truyền thuận tuần tự qua các lớp con."""
        out = x
        for layer in self._layers:
            out = layer.forward(out)
        return out

    def backward(self, grad_out: Tensor) -> Tensor:
        """Lan truyền ngược theo thứ tự đảo ngược qua các lớp con."""
        grad = grad_out
        for layer in reversed(self._layers):
            grad = layer.backward(grad)
        return grad

    def parameters(self) -> list[Parameter]:
        """Gom toàn bộ tham số học được của các lớp con."""
        params: list[Parameter] = []
        for layer in self._layers:
            params.extend(layer.parameters())
        return params

    def train(self) -> None:
        """Bật chế độ huấn luyện cho toàn bộ các lớp con."""
        super().train()
        for layer in self._layers:
            layer.train()

    def eval(self) -> None:
        """Bật chế độ suy luận cho toàn bộ các lớp con."""
        super().eval()
        for layer in self._layers:
            layer.eval()
