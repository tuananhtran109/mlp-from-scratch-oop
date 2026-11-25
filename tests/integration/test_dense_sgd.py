"""Kiểm thử tích hợp: Dense thật (Người A) kết hợp SGD (Người B) giảm hàm mất mát."""

import numpy as np

from nnlib.layers.dense import Dense
from nnlib.optim.sgd import SGD
from nnlib.tensor import Tensor


def test_dense_sgd_linear_regression_loss_decrease() -> None:
    """Chứng minh Dense và SGD phối hợp tối ưu hóa làm giảm rõ rệt hàm mất mát MSE."""
    rng = np.random.default_rng(42)

    # 1. Tạo tập dữ liệu tuyến tính nhân tạo: y = 2*x1 - 3*x2 + 0.5
    n_samples = 40
    x_data = rng.standard_normal((n_samples, 2)).astype(np.float32)
    w_true = np.array([[2.0], [-3.0]], dtype=np.float32)
    b_true = np.array([0.5], dtype=np.float32)
    y_true = x_data @ w_true + b_true

    # 2. Khởi tạo Layer Dense của A và SGD của B
    dense = Dense(in_features=2, out_features=1)
    sgd = SGD(lr=0.05, momentum=0.9)

    x_tensor = Tensor(x_data)

    losses: list[float] = []

    # 3. Huấn luyện 40 bước
    for _ in range(40):
        # Forward
        y_pred = dense.forward(x_tensor)

        # Tính hàm mất mát MSE
        diff = y_pred.data - y_true
        mse = float(np.mean(diff**2))
        losses.append(mse)

        # Gradient của MSE đối với y_pred (đã chia cho N theo chuẩn của hợp đồng)
        grad_out = Tensor((2.0 / n_samples) * diff)

        # Backward tích lũy gradient vào dense.parameters()
        dense.backward(grad_out)

        # Cập nhật trọng số và xóa gradient
        sgd.step(dense.parameters())
        sgd.zero_grad(dense.parameters())

    # 4. Kiểm chứng: Loss giảm mạnh (ít nhất 90% sau 40 bước lặp)
    initial_loss = losses[0]
    final_loss = losses[-1]

    assert (
        final_loss < initial_loss * 0.1
    ), f"Loss không giảm đủ: ban đầu {initial_loss:.4f}, kết thúc {final_loss:.4f}"

    # Kiểm tra nghiệm gần đúng
    w_est = dense.weights.data
    b_est = dense.bias.data
    assert np.allclose(w_est, w_true, atol=0.25)
    assert np.allclose(b_est, b_true, atol=0.25)
