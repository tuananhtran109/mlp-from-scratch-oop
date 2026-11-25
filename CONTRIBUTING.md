# Quy ước và Hợp đồng phát triển dự án (CONTRIBUTING)

## 1. Quy ước kỹ thuật và lập trình
- **Chiều dữ liệu & Biểu diễn nhãn**:
  - Dữ liệu mạng nơ-ron tuân thủ chuẩn batch-first dạng `(N, đặc_trưng)`.
  - Trong `MnistDataset`: Dữ liệu ảnh `_images` được chuẩn hóa về `[0.0, 1.0]` (chia `255.0` một lần duy nhất tại dataset), lưu dạng `np.ndarray` kiểu `float32` shape `(N, 784)`. Nhãn `_labels` lưu dạng mảng số nguyên `(N,)` kiểu `int64` nhằm tiết kiệm bộ nhớ và phục vụ trực tiếp tính ma trận nhầm lẫn (`ConfusionMatrix`).
  - Trong `DataLoader`: Nhãn được chuyển đổi sang vector one-hot `(B, 10)` kiểu `float32` thông qua `OneHotEncoder(num_classes=10)` khi `one_hot=True` (mặc định).
  - Quy ước chuyển đổi `ndarray` $\to$ `Tensor`: `DataLoader` sinh dữ liệu batch dạng mảng thuần NumPy `np.ndarray`. Việc bọc `np.ndarray` thành `Tensor` cho cả dữ liệu đầu vào và nhãn one-hot do `Trainer` (ở Tuần 2) thực hiện trước khi đưa vào mô hình và hàm loss.
- **Gradient theo batch**: Hàm loss trả về gradient đã chia cho kích thước lô `N`, nên các `Layer` tuyệt đối không chia lại cho `N`.
- **Cộng dồn gradient**: Gradient được cộng dồn (`+=`) trong phương thức `backward()`; việc đặt lại gradient về `0` do `IOptimizer.zero_grad()` gọi.
- **Kiểu dữ liệu số thực**: Dùng `float32` khi huấn luyện mô hình và dùng `float64` khi chạy gradient check.
- **Chuẩn mã nguồn**: Tuân thủ PEP 8, khai báo type hints đầy đủ, viết docstring bằng tiếng Việt cho tất cả các lớp public.
- **Quy ước OOP**:
  - Thuộc tính private đặt tên dạng `_ten` và truy cập qua `@property`.
  - Interface kế thừa `abc.ABC` và đặt tên bắt đầu bằng chữ `I` (ví dụ: `ILossFunction`, `IOptimizer`).
  - Các phương thức cài đặt quy tắc nghiệp vụ (R1–R12) cần gắn chú thích `# Rule Rn` để dễ đối chiếu.

## 2. Hợp đồng giao tiếp giữa hai phân hệ
Hai phân hệ (A: lõi `nnlib` và B: huấn luyện + ứng dụng `digitapp`) chỉ giao tiếp thông qua các lớp và interface dưới đây.

**Quy tắc thay đổi hợp đồng**: Sau khi gắn tag `v0.1-contract`, mọi thay đổi về tên thuộc tính hoặc chữ ký phương thức trong mục này bắt buộc phải mở Pull Request gắn nhãn `contract` và được **cả hai thành viên duyệt** trước khi gộp.

```python
# 1. src/nnlib/tensor.py (Người A cài đặt)
class Tensor:  # Bọc ndarray, shape (N, features) theo chuẩn batch-first
    data: np.ndarray          # Truy cập qua @property (lưu nội bộ ở _data)
    shape: tuple[int, ...]    # Truy cập qua @property

class Parameter(Tensor):  # Trọng số học được
    grad: np.ndarray          # Truy cập qua @property (lưu nội bộ ở _grad), do backward tích lũy
    def zero_grad(self) -> None: ...

# 2. src/nnlib/layers/base.py (Người A cài đặt)
class Layer(ABC):
    @abstractmethod
    def forward(self, x: Tensor) -> Tensor: ...
    @abstractmethod
    def backward(self, grad_out: Tensor) -> Tensor: ...
    def parameters(self) -> list[Parameter]: return []
    def train(self) -> None: ...  # Bật chế độ train (Dropout hoạt động)
    def eval(self) -> None: ...   # Bật chế độ suy luận
    def state_dict(self) -> dict[str, np.ndarray]: ...
    def load_state_dict(self, state: dict[str, np.ndarray]) -> None: ...

# 3. src/nnlib/losses/base.py (Người A cài đặt)
class ILossFunction(ABC):
    @abstractmethod
    def forward(self, y_pred: Tensor, y_true: Tensor) -> float: ...
    @abstractmethod
    def backward(self) -> Tensor: ...  # Trả về dL/dy_pred, đã chia cho N

# 4. src/nnlib/optim/base.py (Người B cài đặt)
class IOptimizer(ABC):
    @abstractmethod
    def step(self, params: Sequence[Parameter]) -> None: ...
    @abstractmethod
    def zero_grad(self, params: Sequence[Parameter]) -> None: ...

# 5. Interface: src/nnlib/models/repository.py; cài đặt NpzModelRepository: src/nnlib/io/ (Người A)
class IModelRepository(ABC):
    @abstractmethod
    def save(self, model: Layer, path: str) -> None: ...
    @abstractmethod
    def load(self, path: str) -> Layer: ...

# 6. src/nnlib/training/callbacks.py (Người B cài đặt)
# Ghi chú: Các phương thức mặc định là rỗng để callback con chỉ cần ghi đè sự kiện cần dùng.
class ICallback(ABC):
    def set_trainer(self, trainer: "Trainer") -> None: ...  # Trainer gọi khi đăng ký callback
    def on_train_begin(self, logs: dict | None = None) -> None: ...
    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None: ...
    def on_train_end(self, logs: dict | None = None) -> None: ...
```

**Làm việc song song không chờ nhau**:
- Người B sử dụng `FakeLayer` trong `tests/fakes/` để kiểm thử `SGD`, `Adam`, `Trainer` ngay trong tuần 1–2.
- Người A sử dụng vòng lặp huấn luyện nhỏ trong test để kiểm chứng hội tụ (ví dụ bài toán XOR) mà không phụ thuộc `Trainer`.

## 3. Kiến trúc phân tầng và chiều phụ thuộc package
Kiến trúc gồm 4 tầng (Giao diện -> Ứng dụng -> Lõi -> Hạ tầng); lời gọi chỉ đi một chiều từ trên xuống và được kiểm chứng tự động bằng test kiến trúc `I2`:
- `digitapp.cli` và `digitapp.gui` **chỉ** được import `digitapp.controllers`.
- `nnlib` **không** được import `digitapp`.
- `nnlib.layers` **không** được import `nnlib.optim` hoặc `nnlib.training`.
- Tầng lõi chỉ phụ thuộc vào interface (`IModelRepository`, `IDataset`), không phụ thuộc trực tiếp lớp cài đặt cụ thể ở tầng hạ tầng.

## 4. Quy ước xử lý ngoại lệ, cấu hình và đường dẫn
- **Hệ ngoại lệ thống nhất**:
  - Phân hệ A: `NNError` làm lớp gốc, gồm `ShapeMismatchError`, `InvalidConfigError`, `StateError`, `ModelFileError`.
  - Phân hệ B: `AppError(NNError)` gồm `DataFormatError`, `EmptyCanvasError`.
- **Nguyên tắc xử lý lỗi**:
  - Tầng lõi (`nnlib`) chỉ `raise` ngoại lệ.
  - Tầng điều khiển (`controllers`) mới `try/except` từng loại lỗi cụ thể để chuyển thành thông điệp người dùng (CLI trả mã thoát khác `0`; GUI hiển thị hộp thoại).
  - Tuyệt đối không dùng `except:` trần và không nuốt lỗi; ghi log bằng thư viện `logging` ra thư mục do tệp cấu hình chỉ định.
- **Phân biệt thư mục cấu hình & đường dẫn**:
  - `config/` (ở thư mục gốc dự án): chứa tệp dữ liệu cấu hình `config.json`.
  - `src/digitapp/config/`: package mã nguồn Python chứa lớp `AppConfig` dùng để đọc và kiểm tra `config/config.json`.
  - Mọi đường dẫn tệp phải đi qua hàm `resource_path(relative)` để chạy ổn định cả từ mã nguồn lẫn tệp thực thi đóng gói; không gán cứng đường dẫn tuyệt đối (như `C:\...` hoặc `/home/...`).

## 5. Quy trình Git, nhánh và đóng góp
- **Cấu trúc nhánh**:
  - `main`: lưu các mốc ổn định, gắn tag (`v0.1-contract`, `v0.2`, ..., `v1.0`).
  - `develop`: nhánh tích hợp chung của nhóm.
  - `feature/a-<chủ-đề>` (Người A) và `feature/b-<chủ-đề>` (Người B): nhánh phát triển từng tính năng.
- **Quy ước commit và chỉ tiêu**:
  - Mỗi thành viên đặt `user.name` và `user.email` riêng trên máy cá nhân để lệnh `git shortlog -sn` thống kê chính xác đóng góp.
  - Commit nhỏ, rõ ý theo dạng: `feat(layers): thêm Dropout`, `test(optim): kiểm tra Adam`, `docs: cập nhật README`.
  - Chỉ tiêu tối thiểu: **2 commit/người/tuần** (mục tiêu toàn dự án >= 24 commit, mỗi người >= 12).
- **Pull Request**: Mở PR từ nhánh `feature/...` vào `develop`; người còn lại duyệt trong vòng 24 giờ và gộp bằng **merge commit** (không dùng squash để giữ lịch sử commit cá nhân).
- **Quy tắc `.gitignore`**: Không commit `data/raw/`, `__pycache__/`, `build/`, `dist/`; **vẫn commit** mô hình đã huấn luyện trong `models/`.

## 6. Kiểm tra mã nguồn trước khi mở Pull Request
Cài đặt thư viện (lần đầu hoặc khi có cập nhật):
```bash
pip install -r requirements.txt
```
Chạy đủ 3 lệnh kiểm tra sau và đảm bảo không có lỗi trước khi đẩy mã lên PR:
```bash
pytest
ruff check .
black --check .
```