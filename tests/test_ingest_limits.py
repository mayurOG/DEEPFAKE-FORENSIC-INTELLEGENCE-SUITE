"""Ingest security limits (P1)."""
import pytest
from PIL import Image

from detector.triage import (
    IngestLimitError, enforce_upload_limit, enforce_image_limits, sniff_media_magic,
    enforce_path_size_limit, enforce_duration_limit,
)


def test_upload_limit_rejects_large_payload():
    data = b"\x00" * 1000
    with pytest.raises(IngestLimitError) as ei:
        enforce_upload_limit(data, max_bytes=500)
    assert ei.value.code == "upload_too_large"


def test_upload_limit_allows_small_payload():
    enforce_upload_limit(b"abc", max_bytes=500)


def test_image_pixel_limit():
    img = Image.new("RGB", (100, 100))
    enforce_image_limits(img, max_pixels=10_000)
    with pytest.raises(IngestLimitError) as ei:
        enforce_image_limits(img, max_pixels=100)
    assert ei.value.code == "image_too_large"


def test_sniff_jpeg():
    # minimal JPEG SOI
    assert sniff_media_magic(b"\xff\xd8\xff\xe0" + b"\x00" * 20) == "image"


def test_sniff_unknown_short():
    assert sniff_media_magic(b"ab") is None


def test_path_size_limit(tmp_path):
    f = tmp_path / "big.bin"
    f.write_bytes(b"x" * 200)
    with pytest.raises(IngestLimitError) as ei:
        enforce_path_size_limit(str(f), max_bytes=50)
    assert ei.value.code == "upload_too_large"
    enforce_path_size_limit(str(f), max_bytes=500)


def test_duration_limit_video():
    enforce_duration_limit(30.0, "video", max_video_seconds=60)
    with pytest.raises(IngestLimitError) as ei:
        enforce_duration_limit(200.0, "video", max_video_seconds=120)
    assert ei.value.code == "video_too_long"


def test_duration_limit_audio():
    with pytest.raises(IngestLimitError) as ei:
        enforce_duration_limit(400.0, "audio", max_audio_seconds=300)
    assert ei.value.code == "audio_too_long"
