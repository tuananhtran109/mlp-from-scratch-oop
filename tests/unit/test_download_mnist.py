"""Kiểm thử đơn vị cho các hàm trong scripts/download_mnist.py."""

import gzip
import io
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from scripts.download_mnist import download_file, is_valid_gzip


def test_is_valid_gzip(tmp_path: Path) -> None:
    """Kiểm tra nhận diện tệp gzip hợp lệ và không hợp lệ."""
    gz_file = tmp_path / "valid.gz"
    with gzip.open(gz_file, "wb") as f:
        f.write(b"Hello MNIST")
    assert is_valid_gzip(gz_file) is True

    bad_file = tmp_path / "corrupted.gz"
    bad_file.write_bytes(b"not a gzip content")
    assert is_valid_gzip(bad_file) is False

    empty_file = tmp_path / "empty.gz"
    empty_file.write_bytes(b"")
    assert is_valid_gzip(empty_file) is False

    non_existent = tmp_path / "none.gz"
    assert is_valid_gzip(non_existent) is False


def test_download_file_skip_existing(tmp_path: Path) -> None:
    """Nếu tệp đã tồn tại và hợp lệ thì bỏ qua không tải lại."""
    dest_dir = tmp_path / "raw"
    dest_dir.mkdir()
    target = dest_dir / "test.gz"
    with gzip.open(target, "wb") as f:
        f.write(b"sample data")

    # Gọi download_file, không có mạng vẫn thành công do đã có sẵn
    res = download_file("test.gz", dest_dir)
    assert res == target


def test_download_file_fallback_and_atomic_rename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mô phỏng mirror chính lỗi, tự động chuyển mirror dự phòng và đổi tên atomic."""
    dest_dir = tmp_path / "raw"
    sample_content = b"gzip mock payload"
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb") as f:
        f.write(sample_content)
    valid_gz_bytes = buf.getvalue()

    call_count = 0

    def mock_urlopen(req: urllib.request.Request, timeout: int = 30):
        nonlocal call_count
        call_count += 1
        url = req.full_url
        if "storage.googleapis.com" in url:
            # Mirror chính giả lập lỗi 503
            raise urllib.error.HTTPError(url, 503, "Service Unavailable", {}, None)  # type: ignore[arg-type]
        # Mirror dự phòng thành công
        return io.BytesIO(valid_gz_bytes)

    monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)

    saved_path = download_file("test-file.gz", dest_dir)
    assert saved_path.is_file()
    assert is_valid_gzip(saved_path) is True
    assert call_count == 2

    # Tệp tạm .tmp không còn tồn tại
    assert not (dest_dir / "test-file.gz.tmp").exists()


def test_download_file_both_mirrors_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cả hai mirror đều thất bại phải ném RuntimeError có thông điệp rõ ràng."""
    dest_dir = tmp_path / "raw"

    def mock_urlopen(req: urllib.request.Request, timeout: int = 30):
        raise urllib.error.URLError("Network unreachable")

    monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)

    with pytest.raises(RuntimeError) as exc_info:
        download_file("failing-file.gz", dest_dir)
    assert "Không thể tải tệp 'failing-file.gz'" in str(exc_info.value)
