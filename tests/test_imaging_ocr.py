from datetime import datetime

from PIL import Image

from conftest import make_screenshot
from shotvault.imaging import (
    device_hint,
    dhash,
    for_model,
    hamming,
    has_screenshot_marker,
    looks_like_screenshot,
    open_image,
    taken_at,
)
from shotvault.ocr import reading_order


def test_screenshot_marker_and_names(tmp_path):
    marked = make_screenshot(tmp_path / "IMG_0001.PNG", ["hello"])
    plain = make_screenshot(tmp_path / "photo.png", ["hello"], marker=False)
    assert has_screenshot_marker(marked.read_bytes())
    assert not has_screenshot_marker(plain.read_bytes())
    assert looks_like_screenshot("Captura de Tela 2025-03-14 às 21.07.12.png")
    assert looks_like_screenshot("Screenshot 2025-03-14 at 21.07.12.png")
    assert not looks_like_screenshot("IMG_1234.HEIC")


def test_dhash_survives_reencoding_but_separates_content(tmp_path):
    a = make_screenshot(tmp_path / "a.png", ["one", "two", "three"])
    b = make_screenshot(tmp_path / "b.png", ["completely", "different"], color="black")
    image = open_image(a.read_bytes())
    image.convert("RGB").save(tmp_path / "a.jpg", quality=70)
    ha, hb = dhash(image), dhash(open_image(b.read_bytes()))
    hj = dhash(Image.open(tmp_path / "a.jpg"))
    assert len(ha) == 64
    assert hamming(ha, hj) <= 4
    # flat screenshot backgrounds keep hashes closer; OCR text confirms duplicates
    assert hamming(ha, hb) > 10


def test_taken_at_from_localised_filename():
    assert taken_at(b"", "Captura de Tela 2025-03-14 às 21.07.12.png") == datetime(2025, 3, 14, 21, 7, 12)
    assert taken_at(b"", "IMG_1.PNG", fallback=datetime(2020, 1, 1)) == datetime(2020, 1, 1)


def test_device_hint():
    assert device_hint(1179, 2556) == "phone"
    assert device_hint(1179, 9000).startswith("phone (scrolling")
    assert device_hint(3024, 1964) == "computer"


def test_for_model_downscales_to_jpeg(tmp_path):
    path = make_screenshot(tmp_path / "s.png", ["x"])
    data = for_model(path, max_pixels=200_000, max_side=2048)
    image = Image.open(__import__("io").BytesIO(data))
    assert image.format == "JPEG"
    assert image.width * image.height <= 200_000


def test_reading_order_groups_rows():
    boxes = [
        ("world", 0.9, 0.5, 0.10, 0.2, 0.05),
        ("Hello", 0.9, 0.1, 0.11, 0.2, 0.05),
        ("second line", 0.9, 0.1, 0.30, 0.5, 0.05),
    ]
    assert reading_order(boxes) == "Hello  world\nsecond line"
