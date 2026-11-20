"""Lớp Tensor và Parameter biểu diễn dữ liệu và trọng số học được."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, DTypeLike

from nnlib.exceptions import InvalidConfigError, ShapeMismatchError


class Tensor:
    """Lớp bọc mảng NumPy theo chuẩn batch-first (N, đặc_trưng)."""

    # Cho phép NumPy nhường quyền xử lý toán tử hai ngôi (vd: ndarray * Tensor)
    __array_ufunc__: None = None

    def __init__(
        self,
        data: ArrayLike | Tensor,
        dtype: DTypeLike | None = None,
        copy: bool = False,
    ) -> None:
        """Khởi tạo Tensor từ mảng số hoặc một Tensor khác.

        Quy ước bộ nhớ:
            Mặc định ``copy=False`` không sao chép mảng NumPy đầu vào để tránh
            tốn bộ nhớ. Khi cần lưu cache độc lập (ví dụ trong ``Dense``), hãy
            truyền ``copy=True`` hoặc gọi ``.copy()``.

        Quy ước kiểu dữ liệu (dtype) khi ``dtype=None``:
            - Nếu đầu vào là ``Tensor`` hoặc ``np.ndarray`` kiểu số thực
              (``float32``, ``float64``, ...), giữ nguyên kiểu dữ liệu đó.
            - Các trường hợp còn lại (list/tuple/scalar Python hoặc mảng số
              nguyên) mặc định chuyển sang ``np.float32``.
        """
        is_array_like_typed = isinstance(data, (Tensor, np.ndarray))
        raw = data.data if isinstance(data, Tensor) else data

        try:
            arr = np.asarray(raw, dtype=dtype)
        except (TypeError, ValueError) as exc:
            raise InvalidConfigError(
                "Dữ liệu khởi tạo Tensor không thể chuyển sang mảng số."
            ) from exc

        if not np.issubdtype(arr.dtype, np.number):
            raise InvalidConfigError(
                f"Tensor chỉ chấp nhận kiểu dữ liệu số, nhận được {arr.dtype}."
            )

        if dtype is None:
            if not is_array_like_typed or not np.issubdtype(arr.dtype, np.floating):
                arr = arr.astype(np.float32)

        if copy and arr is raw:
            arr = arr.copy()

        self._data: np.ndarray = arr

    @property
    def data(self) -> np.ndarray:
        """Trả về mảng NumPy bên trong Tensor."""
        return self._data

    @data.setter
    def data(self, value: np.ndarray) -> None:
        """Cập nhật mảng dữ liệu (bắt buộc giữ nguyên shape)."""
        arr = np.asarray(value, dtype=self._data.dtype)
        if arr.shape != self._data.shape:
            raise ShapeMismatchError(
                f"Không thể gán dữ liệu shape {arr.shape} cho Tensor "
                f"shape {self._data.shape}."
            )
        self._data = arr

    @property
    def shape(self) -> tuple[int, ...]:
        """Trả về kích thước (shape) của Tensor."""
        return self._data.shape

    @property
    def dtype(self) -> np.dtype:
        """Trả về kiểu dữ liệu phần tử của Tensor."""
        return self._data.dtype

    @property
    def ndim(self) -> int:
        """Trả về số chiều của Tensor."""
        return self._data.ndim

    @property
    def T(self) -> Tensor:
        """Trả về Tensor chuyển vị."""
        return Tensor(self._data.T)

    def check_shape(self, expected: tuple[int | None, ...]) -> None:
        """Kiểm tra shape khớp mẫu expected (giá trị None bỏ qua chiều đó)."""
        if len(self._data.shape) != len(expected):
            raise ShapeMismatchError(
                f"Kỳ vọng Tensor {len(expected)} chiều {expected}, "
                f"nhận được {self._data.shape}."
            )
        for actual_dim, exp_dim in zip(self._data.shape, expected, strict=True):
            if exp_dim is not None and actual_dim != exp_dim:
                raise ShapeMismatchError(
                    f"Kỳ vọng shape khớp {expected}, nhận được {self._data.shape}."
                )

    def copy(self) -> Tensor:
        """Tạo bản sao độc lập của Tensor."""
        return Tensor(self._data.copy())

    def _extract_operand(
        self, other: Tensor | ArrayLike
    ) -> np.ndarray | int | float | np.number:
        """Trích xuất mảng NumPy hoặc số vô hướng từ toán hạng còn lại."""
        if isinstance(other, Tensor):
            return other.data
        if isinstance(other, (int, float, np.number)) and not isinstance(other, bool):
            return other
        return np.asarray(other)

    def __add__(self, other: Tensor | ArrayLike) -> Tensor:
        """Cộng hai Tensor hoặc Tensor với mảng/hằng số."""
        try:
            return Tensor(self._data + self._extract_operand(other))
        except ValueError as exc:
            raise ShapeMismatchError(f"Lỗi cộng Tensor: {exc}") from exc

    def __radd__(self, other: ArrayLike) -> Tensor:
        """Cộng từ bên phải (hằng số/mảng + Tensor)."""
        return self.__add__(other)

    def __sub__(self, other: Tensor | ArrayLike) -> Tensor:
        """Trừ hai Tensor hoặc Tensor với mảng/hằng số."""
        try:
            return Tensor(self._data - self._extract_operand(other))
        except ValueError as exc:
            raise ShapeMismatchError(f"Lỗi trừ Tensor: {exc}") from exc

    def __rsub__(self, other: ArrayLike) -> Tensor:
        """Trừ từ bên phải (hằng số/mảng - Tensor)."""
        try:
            return Tensor(self._extract_operand(other) - self._data)
        except ValueError as exc:
            raise ShapeMismatchError(f"Lỗi trừ Tensor: {exc}") from exc

    def __mul__(self, other: Tensor | ArrayLike) -> Tensor:
        """Nhân từng phần tử (element-wise)."""
        try:
            return Tensor(self._data * self._extract_operand(other))
        except ValueError as exc:
            raise ShapeMismatchError(f"Lỗi nhân từng phần tử: {exc}") from exc

    def __rmul__(self, other: ArrayLike) -> Tensor:
        """Nhân từng phần tử từ bên phải (hằng số/mảng * Tensor)."""
        return self.__mul__(other)

    def __matmul__(self, other: Tensor | ArrayLike) -> Tensor:
        """Nhân ma trận (matrix multiplication)."""
        try:
            return Tensor(self._data @ self._extract_operand(other))
        except ValueError as exc:
            raise ShapeMismatchError(f"Lỗi nhân ma trận: {exc}") from exc

    def __rmatmul__(self, other: ArrayLike) -> Tensor:
        """Nhân ma trận từ bên phải (mảng @ Tensor)."""
        try:
            return Tensor(self._extract_operand(other) @ self._data)
        except ValueError as exc:
            raise ShapeMismatchError(f"Lỗi nhân ma trận: {exc}") from exc

    def __neg__(self) -> Tensor:
        """Đổi dấu toàn bộ phần tử trong Tensor."""
        return Tensor(-self._data)

    def __repr__(self) -> str:
        """Biểu diễn chuỗi phục vụ gỡ lỗi."""
        return f"Tensor(shape={self.shape}, dtype={self.dtype})"


class Parameter(Tensor):
    """Trọng số học được của Layer, kế thừa Tensor và lưu thêm gradient."""

    def __init__(
        self,
        data: ArrayLike | Tensor,
        name: str = "",
        dtype: DTypeLike | None = None,
        copy: bool = False,
    ) -> None:
        """Khởi tạo Parameter với mảng số thực, tên và gradient bằng 0."""
        super().__init__(data, dtype=dtype, copy=copy)
        if not np.issubdtype(self._data.dtype, np.floating):
            raise InvalidConfigError("Parameter chỉ chấp nhận kiểu số thực.")
        self._name: str = name
        self._grad: np.ndarray = np.zeros_like(self._data)

    @property
    def name(self) -> str:
        """Trả về tên tham số phục vụ lưu/nạp mô hình."""
        return self._name

    @property
    def grad(self) -> np.ndarray:
        """Trả về mảng gradient tích lũy cùng shape với data."""
        return self._grad

    @grad.setter
    def grad(self, value: np.ndarray) -> None:
        """Cập nhật gradient (bắt buộc cùng shape với data)."""
        arr = np.asarray(value, dtype=self._data.dtype)
        if arr.shape != self._data.shape:
            raise ShapeMismatchError(
                f"Gradient có shape {arr.shape} không khớp tham số "
                f"shape {self._data.shape}."
            )
        self._grad = arr

    def zero_grad(self) -> None:
        """Đặt toàn bộ gradient tích lũy về 0."""
        self._grad.fill(0.0)

    def __repr__(self) -> str:
        """Biểu diễn chuỗi phục vụ gỡ lỗi."""
        return (
            f"Parameter(name={self._name!r}, shape={self.shape}, "
            f"dtype={self.dtype})"
        )
