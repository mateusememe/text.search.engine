"""Image helpers: screenshot detection, hashing, resizing for models and the vault."""

from __future__ import annotations

import hashlib
import io
import re
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageOps

try:  # HEIC/HEIF support (iPhone camera roll); optional
    from pillow_heif import register_heif_opener

    register_heif_opener()
except ImportError:  # pragma: no cover - depends on optional extra
    pass

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".heic", ".heif", ".webp"}

# Localised names macOS / Windows give screenshot files.
SCREENSHOT_NAME = re.compile(
    r"(screen ?shot|captura de (tela|pantalla)|capture d.(é|e)cran|bildschirmfoto|"
    r"schermata|schermafbeelding|zrzut ekranu|снимок экрана|スクリーンショット|截屏|屏幕截图|스크린샷)",
    re.IGNORECASE,
)

# iOS and macOS write "Screenshot" into EXIF/XMP UserComment.
_MARKER = re.compile(rb">Screenshot<|UserComment=[\"']Screenshot|ASCII\x00\x00\x00Screenshot")
_XMP_DATE = re.compile(rb"(?:photoshop:DateCreated|xmp:CreateDate|exif:DateTimeOriginal)[=>\"']+([0-9T:\-+.Z]{10,30})")
_NAME_DATE = re.compile(r"(\d{4})-(\d{2})-(\d{2})\D{1,8}(\d{1,2})[.:h](\d{2})[.:m](\d{2})")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def has_screenshot_marker(data: bytes) -> bool:
    return bool(_MARKER.search(data[: 512 * 1024]))


def looks_like_screenshot(name: str, data: bytes | None = None) -> bool:
    if SCREENSHOT_NAME.search(name):
        return True
    return bool(data) and has_screenshot_marker(data)


def open_image(data: bytes) -> Image.Image:
    image = Image.open(io.BytesIO(data))
    image.load()
    return ImageOps.exif_transpose(image)


def dhash(image: Image.Image, size: int = 16) -> str:
    """Difference hash (size² bits) as hex; robust to re-encoding and resizing."""
    gray = image.convert("L").resize((size + 1, size), Image.Resampling.LANCZOS)
    pixels = gray.tobytes()
    bits = 0
    for row in range(size):
        for col in range(size):
            left = pixels[row * (size + 1) + col]
            right = pixels[row * (size + 1) + col + 1]
            bits = (bits << 1) | (left > right)
    return f"{bits:0{size * size // 4}x}"


def hamming(a: str, b: str) -> int:
    return (int(a, 16) ^ int(b, 16)).bit_count()


def taken_at(data: bytes, name: str, fallback: datetime | None = None) -> datetime | None:
    """Best-effort capture time from EXIF, XMP, then the file name."""
    try:
        exif = Image.open(io.BytesIO(data)).getexif()
        raw = exif.get_ifd(0x8769).get(36867) or exif.get(306)
        if raw:
            return datetime.strptime(str(raw).strip("\x00"), "%Y:%m:%d %H:%M:%S")
    except Exception:
        pass
    match = _XMP_DATE.search(data[: 512 * 1024])
    if match:
        try:
            value = match.group(1).decode().replace("Z", "+00:00")
            parsed = datetime.fromisoformat(value)
            return parsed.astimezone().replace(tzinfo=None) if parsed.tzinfo else parsed
        except ValueError:
            pass
    match = _NAME_DATE.search(name)
    if match:
        try:
            return datetime(*(int(g) for g in match.groups()))
        except ValueError:
            pass
    return fallback


def device_hint(width: int, height: int) -> str:
    if not width or not height:
        return "unknown device"
    ratio = max(width, height) / min(width, height)
    if height > width and ratio >= 1.7:
        return "phone" if ratio < 3 else "phone (scrolling capture)"
    if height > width:
        return "tablet"
    return "computer" if ratio < 2.5 else "wide capture"


def fit(image: Image.Image, max_pixels: int, max_side: int) -> Image.Image:
    width, height = image.size
    scale = min(1.0, (max_pixels / (width * height)) ** 0.5, max_side / max(width, height))
    if scale >= 1.0:
        return image
    return image.resize((max(1, int(width * scale)), max(1, int(height * scale))), Image.Resampling.LANCZOS)


def _flatten(image: Image.Image) -> Image.Image:
    if image.mode in ("RGBA", "LA", "P"):
        image = image.convert("RGBA")
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.getchannel("A"))
        return background
    return image.convert("RGB")


def for_model(path: Path, max_pixels: int, max_side: int) -> bytes:
    """JPEG sized for a vision model; OCR already carries the exact text."""
    image = _flatten(fit(open_image(path.read_bytes()), max_pixels, max_side))
    out = io.BytesIO()
    image.save(out, "JPEG", quality=90)
    return out.getvalue()


def write_attachment(src: Path, dest: Path, fmt: str, quality: int, max_side: int) -> None:
    # Cap total pixels rather than height so long scrolling captures stay legible.
    image = fit(open_image(src.read_bytes()), max_side * max_side // 2, 16000)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "jpeg":
        _flatten(image).save(dest, "JPEG", quality=quality, optimize=True)
    elif fmt == "png":
        image.save(dest, "PNG", optimize=True)
    else:
        image.save(dest, "WEBP", quality=quality, method=6)
