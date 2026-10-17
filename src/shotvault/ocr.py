"""OCR engines. Apple Vision on macOS (fast, on the Neural Engine), Tesseract elsewhere.

OCR runs before the vision model: it gives exact text, so the model can spend
its effort interpreting instead of transcribing (and hallucinates less).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Protocol

from PIL import Image

from shotvault.config import OCRConfig

# (text, confidence, x, y_top, width, height) with coordinates normalised to 0..1.
Box = tuple[str, float, float, float, float, float]


class OCREngine(Protocol):
    name: str

    def recognize(self, path: Path) -> str: ...


def reading_order(boxes: list[Box]) -> str:
    """Group word/line boxes into rows (top to bottom, left to right)."""
    rows: list[list[Box]] = []
    for box in sorted(boxes, key=lambda b: (b[3], b[2])):
        center = box[3] + box[5] / 2
        if rows:
            last = rows[-1]
            ref = sum(b[3] + b[5] / 2 for b in last) / len(last)
            height = max(b[5] for b in last)
            if abs(center - ref) < max(height, box[5]) * 0.5:
                last.append(box)
                continue
        rows.append([box])
    return "\n".join("  ".join(b[0] for b in sorted(row, key=lambda b: b[2])) for row in rows).strip()


class VisionOCR:
    """Apple Vision via ocrmac. `livetext` uses the newer Live Text analyser (macOS 13+)."""

    def __init__(self, languages: list[str], framework: str = "vision"):
        from ocrmac import ocrmac  # noqa: F401  (fail fast if missing)

        self.languages = languages or None
        self.framework = framework
        self.name = f"apple-{framework}"

    def recognize(self, path: Path) -> str:
        from ocrmac import ocrmac

        image = Image.open(path)
        kwargs = {"framework": self.framework, "language_preference": self.languages}
        if self.framework == "vision":
            kwargs["recognition_level"] = "accurate"
        try:
            results = ocrmac.OCR(image, **kwargs).recognize()
        except ValueError:  # a language Vision doesn't know on this macOS version
            kwargs["language_preference"] = None
            results = ocrmac.OCR(image, **kwargs).recognize()
        boxes: list[Box] = []
        for text, confidence, (x, y, w, h) in results:
            # Vision uses a bottom-left origin.
            boxes.append((str(text), float(confidence), x, 1.0 - y - h, w, h))
        return reading_order(boxes)


_TESSERACT_LANGS = {"en": "eng", "pt": "por", "es": "spa", "fr": "fra", "de": "deu", "it": "ita", "ja": "jpn"}


class TesseractOCR:
    name = "tesseract"

    def __init__(self, languages: list[str]):
        import pytesseract

        self._tess = pytesseract
        codes = [_TESSERACT_LANGS.get(lang.split("-")[0].lower()) for lang in languages]
        available = set(pytesseract.get_languages(config=""))
        self.lang = "+".join(c for c in codes if c and c in available) or "eng"

    def recognize(self, path: Path) -> str:
        return self._tess.image_to_string(Image.open(path), lang=self.lang).strip()


class NoOCR:
    name = "none"

    def recognize(self, path: Path) -> str:
        return ""


def make_ocr(cfg: OCRConfig) -> OCREngine:
    engine = cfg.engine
    if engine == "auto":
        if sys.platform == "darwin":
            try:
                return VisionOCR(cfg.languages)
            except ImportError:
                pass
        try:
            return TesseractOCR(cfg.languages)
        except Exception:
            return NoOCR()
    if engine in ("vision", "livetext"):
        return VisionOCR(cfg.languages, framework=engine)
    if engine == "tesseract":
        return TesseractOCR(cfg.languages)
    return NoOCR()
