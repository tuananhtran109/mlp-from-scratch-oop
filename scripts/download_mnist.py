"""Script tải bộ dữ liệu MNIST định dạng IDX nén (.gz) về data/raw/ an toàn."""

from __future__ import annotations

import argparse
import gzip
import sys
import urllib.error
import urllib.request
from pathlib import Path

# Danh sách 4 tệp MNIST IDX chuẩn
MNIST_FILES: tuple[str, ...] = (
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
)

# Nguồn tải chính và dự phòng
PRIMARY_MIRROR: str = "https://storage.googleapis.com/cvdf-datasets/mnist/"
BACKUP_MIRROR: str = "https://ossci-datasets.s3.amazonaws.com/mnist/"


def is_valid_gzip(file_path: Path) -> bool:
    """Kiểm tra xem tệp có tồn tại và là tệp gzip hợp lệ hay không."""
    if not file_path.is_file() or file_path.stat().st_size == 0:
        return False
    try:
        with gzip.open(file_path, "rb") as f:
            _ = f.read(64)
        return True
    except (OSError, gzip.BadGzipFile):
        return False


def download_file(
    filename: str,
    dest_dir: Path,
    primary_mirror: str = PRIMARY_MIRROR,
    backup_mirror: str = BACKUP_MIRROR,
) -> Path:
    """Tải một tệp về dest_dir với cơ chế atomic rename và mirror dự phòng."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    target_path = dest_dir / filename

    # 1. Bỏ qua nếu tệp đã tồn tại và hợp lệ
    if is_valid_gzip(target_path):
        print(f"[ĐÃ CÓ] {filename} đã tồn tại hợp lệ trong {dest_dir}, bỏ qua.")
        return target_path

    tmp_path = dest_dir / f"{filename}.tmp"
    mirrors = [primary_mirror, backup_mirror]
    download_success = False

    for mirror in mirrors:
        url = mirror.rstrip("/") + "/" + filename
        print(f"[ĐANG TẢI] Đang kết nối tới: {url} ...")
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "Mozilla/5.0 (Python urllib)"}
            )
            with (
                urllib.request.urlopen(req, timeout=30) as response,
                open(tmp_path, "wb") as out_file,
            ):
                total_bytes = 0
                while True:
                    chunk = response.read(65536)
                    if not chunk:
                        break
                    out_file.write(chunk)
                    total_bytes += len(chunk)
            print(f"[THÀNH CÔNG] Đã nhận {total_bytes} bytes từ {mirror}")
            download_success = True
            break
        except (
            urllib.error.URLError,
            urllib.error.HTTPError,
            TimeoutError,
            OSError,
        ) as exc:
            print(f"[CẢNH BÁO] Không thể tải từ {mirror}: {exc}")
            if tmp_path.exists():
                tmp_path.unlink()

    if not download_success:
        raise RuntimeError(
            f"Lỗi: Không thể tải tệp '{filename}' từ cả mirror chính và dự phòng. "
            f"Vui lòng kiểm tra lại kết nối mạng."
        )

    # 2. Kiểm tra tính toàn vẹn của tệp gzip trước khi đổi tên chính thức
    if not is_valid_gzip(tmp_path):
        if tmp_path.exists():
            tmp_path.unlink()
        raise RuntimeError(
            f"Lỗi: Tệp tạm '{tmp_path.name}' không phải tệp gzip hợp lệ."
        )

    # 3. Đổi tên atomic sang tệp đích
    tmp_path.replace(target_path)
    print(f"[HOÀN THÀNH] Đã lưu tệp chính thức: {target_path}")
    return target_path


def download_mnist(dest_dir: Path) -> list[Path]:
    """Tải toàn bộ 4 tệp của bộ dữ liệu MNIST về dest_dir."""
    print(f"=== Bắt đầu tải bộ dữ liệu MNIST về thư mục: {dest_dir} ===")
    results: list[Path] = []
    for filename in MNIST_FILES:
        saved_path = download_file(filename, dest_dir)
        results.append(saved_path)
    print("=== Hoàn tất quá trình tải bộ dữ liệu MNIST! ===")
    return results


def main() -> None:
    """Điểm vào dòng lệnh cho script download_mnist."""
    parser = argparse.ArgumentParser(description="Tải dữ liệu MNIST IDX vào data/raw/")
    default_dir = Path(__file__).resolve().parents[1] / "data" / "raw"
    parser.add_argument(
        "--dest-dir",
        type=Path,
        default=default_dir,
        help="Thư mục lưu dữ liệu thô (mặc định: data/raw/)",
    )
    args = parser.parse_args()

    try:
        download_mnist(args.dest_dir)
    except RuntimeError as exc:
        print(f"\n[LỖI]: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
