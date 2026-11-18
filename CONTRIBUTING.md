# Quy ước phát triển dự án (CONTRIBUTING)

## 1. Quy ước kỹ thuật
- **Chiều dữ liệu**: Tensor batch-first dạng `(N, đặc_trưng)`; nhãn phân loại dạng one-hot `(N, 10)`.
- **Gradient theo batch**: Hàm loss trả về gradient đã chia cho `N`, nên các `Layer` không chia lại cho `N`.
- **Kiểu dữ liệu**: `float32` khi huấn luyện, `float64` khi chạy gradient check.
- **Chuẩn mã nguồn**: PEP 8, có type hints đầy đủ, docstring tiếng Việt cho mọi lớp public.
- **Thuộc tính & Interface**: Thuộc tính private đặt dạng `_ten` và truy cập qua `@property`; interface bắt đầu bằng `I` kế thừa `abc.ABC`.
- **Ngoại lệ**: Mọi lỗi nghiệp vụ/lõi đều kế thừa từ `NNError`.

## 2. Quy trình Git
- Nhánh `main`: lưu các mốc ổn định (`v0.1-contract`, `v0.2`, ..., `v1.0`).
- Nhánh `develop`: nhánh tích hợp chung.
- Nhánh tính năng: `feature/a-<chủ-đề>` (Người A) và `feature/b-<chủ-đề>` (Người B).
- Mở Pull Request vào `develop`, người còn lại duyệt trong 24 giờ và gộp bằng **merge commit** (không squash).
## 3. Ph�n bi?t thu m?c c?u h�nh
- `config/` (? thu m?c g?c d? �n): ch?a t?p d? li?u c?u h�nh `config.json`.
- `src/digitapp/config/`: package m� ngu?n Python ch?a l?p `AppConfig` d�ng d? d?c v� ki?m tra t?p `config/config.json`.
