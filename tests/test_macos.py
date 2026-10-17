"""Runs only on a Mac with the `mac` extra: real Apple Vision OCR."""

import sys

import pytest
from PIL import Image, ImageDraw

pytestmark = pytest.mark.skipif(sys.platform != "darwin", reason="Apple Vision needs macOS")


def test_vision_ocr_reads_lines_in_order(tmp_path):
    pytest.importorskip("ocrmac")
    from shotvault.ocr import VisionOCR

    image = Image.new("RGB", (1179, 1200), "white")
    draw = ImageDraw.Draw(image)
    draw.text((80, 200), "Shopping list", fill="black", font_size=90)
    draw.text((80, 500), "Oat milk and coffee", fill="black", font_size=90)
    path = tmp_path / "ocr.png"
    image.save(path)
    lines = VisionOCR(["en-US", "pt-BR"]).recognize(path).lower().splitlines()
    assert "shopping" in lines[0]
    assert "coffee" in lines[-1]
