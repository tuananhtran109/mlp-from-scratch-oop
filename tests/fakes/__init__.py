"""Package các lớp giả lập (fakes) phục vụ kiểm thử độc lập."""

from tests.fakes.fake_layer import FakeLayer
from tests.fakes.fake_loss import FakeMSELoss
from tests.fakes.fake_sequential import FakeSequential

__all__ = ["FakeLayer", "FakeMSELoss", "FakeSequential"]
