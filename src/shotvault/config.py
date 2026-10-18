"""Configuration: a single TOML file, validated with pydantic."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

CONFIG_ENV = "SHOTVAULT_CONFIG"
DEFAULT_CONFIG_PATH = Path("~/.config/shotvault/config.toml").expanduser()


class ConfigError(RuntimeError):
    pass


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PhotosSource(_Section):
    """Apple Photos library (includes everything synced from iCloud Photos)."""

    enabled: bool = True
    library: str | None = None  # None = the system library
    download_missing: bool = True  # fetch originals that only live in iCloud
    use_photokit: bool = False  # faster than AppleScript, needs Terminal.app
    include_hidden: bool = False
    since: str | None = None  # YYYY-MM-DD, ignore older screenshots


class IPhoneSource(_Section):
    """iPhone connected over USB (read-only, nothing is deleted from the phone)."""

    enabled: bool = True
    serial: str | None = None  # None = first connected device
    accept_all_png: bool = False  # import PNGs even without the screenshot marker


class FolderSource(_Section):
    """Plain folders, e.g. macOS screenshots on the Desktop or an AirDrop inbox."""

    enabled: bool = True
    paths: list[str] = Field(default_factory=lambda: ["~/Desktop"])
    recursive: bool = False
    only_screenshots: bool = True


class Sources(_Section):
    photos: PhotosSource = Field(default_factory=PhotosSource)
    iphone: IPhoneSource = Field(default_factory=IPhoneSource)
    folders: FolderSource = Field(default_factory=FolderSource)


class OCRConfig(_Section):
    engine: Literal["auto", "vision", "livetext", "tesseract", "none"] = "auto"
    languages: list[str] = Field(default_factory=lambda: ["en-US", "pt-BR"])
    max_chars: int = 6000  # OCR text passed to the model


class LLMConfig(_Section):
    provider: Literal["ollama", "openai"] = "ollama"  # "openai" = LM Studio or any OpenAI-compatible server
    base_url: str | None = None
    api_key: str | None = None
    vision_model: str = "qwen2.5vl:7b"
    text_model: str | None = None  # topic synthesis + ask; defaults to vision_model
    embed_model: str | None = "bge-m3"  # None disables semantic search
    temperature: float = 0.1
    num_ctx: int = 8192
    max_tokens: int = 2048
    max_image_pixels: int = 1_600_000
    max_image_side: int = 2048
    timeout: float = 600
    keep_alive: str = "15m"
    think: bool | None = None  # only sent when set; for reasoning models in Ollama

    @property
    def url(self) -> str:
        if self.base_url:
            return self.base_url.rstrip("/")
        return "http://localhost:11434" if self.provider == "ollama" else "http://localhost:1234/v1"

    @property
    def chat_model(self) -> str:
        return self.text_model or self.vision_model


class AttachmentsConfig(_Section):
    mode: Literal["copy", "none"] = "copy"
    format: Literal["webp", "jpeg", "png"] = "webp"
    quality: int = 80
    max_side: int = 2400
    embed_width: int = 420


class PrivacyConfig(_Section):
    # redact: note without image/text, details stay in the local database
    # skip:   no note at all
    # keep:   treat like any other screenshot
    sensitive: Literal["redact", "skip", "keep"] = "redact"


class KnowledgeConfig(_Section):
    topics: bool = True
    topic_min_shots: int = 5
    max_topics: int = 80
    topic_synthesis: bool = True
    topic_max_entries: int = 40
    entities: bool = True
    entity_min_mentions: int = 2
    monthly: bool = True
    on_this_day: bool = True
    rediscover: int = 5


class GitConfig(_Section):
    commit: bool = False
    push: bool = False


class Config(_Section):
    vault: Path
    folder: str = "Screenshots"
    data_dir: Path = Path("~/.shotvault")
    language: str = "English"
    sources: Sources = Field(default_factory=Sources)
    ocr: OCRConfig = Field(default_factory=OCRConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    attachments: AttachmentsConfig = Field(default_factory=AttachmentsConfig)
    privacy: PrivacyConfig = Field(default_factory=PrivacyConfig)
    knowledge: KnowledgeConfig = Field(default_factory=KnowledgeConfig)
    git: GitConfig = Field(default_factory=GitConfig)

    @field_validator("vault", "data_dir", mode="before")
    @classmethod
    def _expand(cls, value: str | Path) -> Path:
        return Path(value).expanduser()

    @property
    def root(self) -> Path:
        """Folder inside the vault that shotvault writes to."""
        return self.vault / self.folder

    @property
    def db_path(self) -> Path:
        return self.data_dir / "shotvault.db"

    @property
    def originals_dir(self) -> Path:
        return self.data_dir / "originals"

    @property
    def staging_dir(self) -> Path:
        return self.data_dir / "staging"

    @property
    def logs_dir(self) -> Path:
        return self.data_dir / "logs"


def config_path(explicit: Path | None = None) -> Path:
    if explicit:
        return explicit.expanduser()
    if os.environ.get(CONFIG_ENV):
        return Path(os.environ[CONFIG_ENV]).expanduser()
    local = Path("shotvault.toml")
    if local.exists():
        return local
    return DEFAULT_CONFIG_PATH


def load_config(explicit: Path | None = None) -> Config:
    path = config_path(explicit)
    if not path.exists():
        raise ConfigError(f"No config at {path}. Create one with `shotvault init <path-to-vault>`.")
    try:
        cfg = Config.model_validate(tomllib.loads(path.read_text()))
    except (tomllib.TOMLDecodeError, ValidationError) as exc:
        raise ConfigError(f"Invalid config {path}:\n{exc}") from exc
    return cfg


CONFIG_TEMPLATE = """\
# shotvault configuration — https://github.com/mateusememe/shotvault

vault = "{vault}"          # your Obsidian vault
folder = "Screenshots"     # shotvault only writes inside this folder
data_dir = "~/.shotvault"  # database, originals, logs (stays outside the vault)
language = "English"       # language the notes are written in, e.g. "Português (Brasil)"

[sources.photos]           # Apple Photos = everything in iCloud Photos
enabled = true
download_missing = true    # download originals that are only in iCloud
use_photokit = false       # faster downloads; only works from Terminal.app
include_hidden = false
# since = "2023-01-01"

[sources.iphone]           # iPhone plugged in over USB (unlock it + "Trust")
enabled = true
accept_all_png = false

[sources.folders]          # macOS screenshots land on the Desktop by default
enabled = true
paths = ["~/Desktop"]
only_screenshots = true

[ocr]
engine = "auto"            # auto | vision | livetext | tesseract | none
languages = ["en-US", "pt-BR"]

[llm]
provider = "{provider}"      # ollama | openai (LM Studio)
# base_url = "{base_url}"
vision_model = "{vision_model}"
# text_model = "qwen3:8b"  # used for topic synthesis and `ask`; defaults to vision_model
embed_model = "{embed_model}"
num_ctx = 8192

[attachments]
mode = "copy"              # copy | none (vault stays text-only)
format = "webp"
quality = 80
max_side = 2400

[privacy]
sensitive = "redact"       # redact | skip | keep

[knowledge]
topics = true
topic_min_shots = 5
topic_synthesis = true
entities = true
on_this_day = true

[git]
commit = true              # commit the vault after each run
push = false               # leave pushing to Obsidian Git if you use it
"""
