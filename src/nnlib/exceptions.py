"""Hệ thống ngoại lệ chuẩn cho thư viện mạng nơ-ron nnlib."""


class NNError(Exception):
    """Ngoại lệ cơ sở cho toàn bộ thư viện mạng nơ-ron nnlib."""

    def __init__(self, message: str = "") -> None:
        """Khởi tạo ngoại lệ với thông điệp mô tả lỗi."""
        super().__init__(message)
        self._message: str = message

    @property
    def message(self) -> str:
        """Trả về thông điệp mô tả lỗi."""
        return self._message


class ShapeMismatchError(NNError):
    """Ném ra khi kích thước (shape) của Tensor không khớp yêu cầu."""


class InvalidConfigError(NNError):
    """Ném ra khi tham số khởi tạo hoặc cấu hình không hợp lệ."""


class StateError(NNError):
    """Ném ra khi gọi sai trình tự trạng thái (ví dụ backward trước forward)."""


class ModelFileError(NNError):
    """Ném ra khi lưu/nạp tệp mô hình thất bại hoặc tệp bị hỏng."""
