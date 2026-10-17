from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from shotvault.config import Config
from shotvault.db import Database

PNG_SCREENSHOT_XMP = (
    '<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
    '<rdf:Description xmlns:exif="http://ns.adobe.com/exif/1.0/"><exif:UserComment><rdf:Alt>'
    '<rdf:li xml:lang="x-default">Screenshot</rdf:li></rdf:Alt></exif:UserComment></rdf:Description>'
    "</rdf:RDF></x:xmpmeta>"
)


def make_screenshot(path: Path, lines: list[str], color: str = "white", marker: bool = True) -> Path:
    """A phone-shaped PNG with text, optionally carrying the iOS screenshot marker."""
    from PIL.PngImagePlugin import PngInfo

    image = Image.new("RGB", (590, 1278), color)
    draw = ImageDraw.Draw(image)
    for i, line in enumerate(lines):
        draw.rectangle((30, 100 + i * 90, 30 + 18 * len(line), 150 + i * 90), fill=(40 + i * 30, 60, 90))
        draw.text((40, 110 + i * 90), line, fill="white", font_size=28)
    info = PngInfo()
    if marker:
        info.add_itxt("XML:com.adobe.xmp", PNG_SCREENSHOT_XMP)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, pnginfo=info)
    return path


class FakeOCR:
    """Returns text registered per image content (by sha256)."""

    name = "fake"

    def __init__(self):
        self.texts: dict[str, str] = {}

    def register(self, path: Path, text: str) -> None:
        self.texts[hashlib.sha256(path.read_bytes()).hexdigest()] = text

    def recognize(self, path: Path) -> str:
        return self.texts.get(path.stem, "")


class FakeLLM:
    """Deterministic stand-in for Ollama / LM Studio."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.vision_calls = 0
        self.chat_calls = 0
        self.prompts: list[str] = []

    def vision_json(self, system, prompt, image, schema):
        self.vision_calls += 1
        self.prompts.append(prompt)
        assert image[:2] == b"\xff\xd8", "model input should be JPEG"
        ocr = prompt.split('"""')[1]
        if "rust" in ocr.lower():
            return json.dumps(
                {
                    "description": "A tweet thread about Rust async cancellation.",
                    "kind": "tweet",
                    "source_app": "X",
                    "title": "Tokio select! cancellation gotcha",
                    "summary": "Branches in tokio::select! that lose are dropped mid-await, so futures must be cancellation safe.",
                    "key_points": ["select! drops losing branches", "use cancellation-safe futures"],
                    "why_saved": "Reference for a cancellation bug.",
                    "entities": [{"name": "Tokio", "type": "technology"}, {"name": "rust", "type": "technology"}],
                    "tags": ["#Rust", "async programming", "screenshot"],
                    "urls": ["docs.rs/tokio", "hallucinated.example.com"],
                    "actions": [{"task": "Audit select! usages", "due": "2025-03-20"}],
                    "code": "```rust\ntokio::select! { a = f() => {}, b = g() => {} }\n```",
                    "code_language": "rust",
                    "language": "en",
                    "keep_value": "reference",
                    "sensitive": False,
                    "sensitive_reason": None,
                }
            )
        if "verification code" in ocr.lower():
            return json.dumps(
                {
                    "description": "An SMS with a bank verification code.",
                    "kind": "notification",
                    "title": "Bank code 482913",
                    "summary": "Your code is 482913.",
                    "tags": ["banking"],
                    "keep_value": "ephemeral",
                    "sensitive": True,
                    "sensitive_reason": "one-time code",
                }
            )
        return json.dumps(
            {
                "description": "A recipe card for banana bread.",
                "kind": "recipe",
                "source_app": "Instagram",
                "title": "Banana bread with rust-free tins",  # note: 'rust' only in title
                "summary": "Mash 3 bananas, mix with flour and bake 50 minutes.",
                "key_points": ["3 ripe bananas", "Bake at 180°C for 50 min"],
                "why_saved": "To bake this weekend.",
                "entities": [{"name": "Rust", "type": "concept"}],
                "tags": ["baking", "rust"],
                "actions": ["Buy bananas"],
                "keep_value": "useful",
            }
        )

    def chat(self, system, prompt, stream=False):
        self.chat_calls += 1
        yield "## Overview\nShots about #rust [1].\n\n## Key insights\n- select! drops branches [1, 2]\n"

    def embed(self, texts):
        vectors = []
        for text in texts:
            vec = [0.0] * 32
            for word in re.findall(r"\w+", text.lower()):
                vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % 32] += 1.0
            vectors.append(vec)
        return vectors

    def models(self):
        return [self.cfg.llm.vision_model, "bge-m3:latest"]


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    vault = tmp_path / "vault"
    (vault / ".obsidian").mkdir(parents=True)
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    return Config.model_validate(
        {
            "vault": str(vault),
            "data_dir": str(tmp_path / "data"),
            "sources": {
                "photos": {"enabled": False},
                "iphone": {"enabled": False},
                "folders": {"paths": [str(inbox)]},
            },
            "knowledge": {"topic_min_shots": 2, "entity_min_mentions": 2},
        }
    )


@pytest.fixture
def db(cfg: Config) -> Database:
    database = Database(cfg.db_path)
    yield database
    database.close()
