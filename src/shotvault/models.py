"""The structured knowledge extracted from one screenshot."""

from __future__ import annotations

import copy
import re
import unicodedata
from enum import Enum
from typing import Any

from pydantic import BaseModel, field_validator


class Kind(str, Enum):
    ARTICLE = "article"
    SOCIAL_POST = "social_post"
    CHAT = "chat"
    CODE = "code"
    ERROR = "error"
    TUTORIAL = "tutorial"
    QUOTE = "quote"
    RECIPE = "recipe"
    SHOPPING = "shopping"
    RECEIPT = "receipt"
    FINANCE = "finance"
    TRAVEL = "travel"
    PLACE = "place"
    EVENT = "event"
    MEDIA = "media"
    DESIGN = "design"
    CHART = "chart"
    DOCUMENT = "document"
    NOTIFICATION = "notification"
    MEME = "meme"
    OTHER = "other"


KIND_LABELS = {
    Kind.ARTICLE: "📰 Article",
    Kind.SOCIAL_POST: "💬 Social post",
    Kind.CHAT: "🗨️ Chat",
    Kind.CODE: "💻 Code",
    Kind.ERROR: "🐞 Error",
    Kind.TUTORIAL: "🧭 Tutorial",
    Kind.QUOTE: "❝ Quote",
    Kind.RECIPE: "🍳 Recipe",
    Kind.SHOPPING: "🛍️ Shopping",
    Kind.RECEIPT: "🧾 Receipt",
    Kind.FINANCE: "💰 Finance",
    Kind.TRAVEL: "✈️ Travel",
    Kind.PLACE: "📍 Place",
    Kind.EVENT: "📅 Event",
    Kind.MEDIA: "🎬 Media",
    Kind.DESIGN: "🎨 Design",
    Kind.CHART: "📊 Chart",
    Kind.DOCUMENT: "📄 Document",
    Kind.NOTIFICATION: "🔔 Notification",
    Kind.MEME: "😂 Meme",
    Kind.OTHER: "🗂️ Other",
}

_KIND_SYNONYMS = {
    "tweet": Kind.SOCIAL_POST,
    "post": Kind.SOCIAL_POST,
    "social": Kind.SOCIAL_POST,
    "social_media": Kind.SOCIAL_POST,
    "message": Kind.CHAT,
    "conversation": Kind.CHAT,
    "terminal": Kind.CODE,
    "command": Kind.CODE,
    "map": Kind.PLACE,
    "location": Kind.PLACE,
    "product": Kind.SHOPPING,
    "music": Kind.MEDIA,
    "movie": Kind.MEDIA,
    "book": Kind.MEDIA,
    "video": Kind.MEDIA,
    "ui": Kind.DESIGN,
    "website": Kind.ARTICLE,
    "news": Kind.ARTICLE,
    "bank": Kind.FINANCE,
}


class EntityType(str, Enum):
    PERSON = "person"
    ORGANIZATION = "organization"
    PRODUCT = "product"
    PLACE = "place"
    TECHNOLOGY = "technology"
    CONCEPT = "concept"
    WORK = "work"  # book, film, song, paper, course
    OTHER = "other"


class KeepValue(str, Enum):
    REFERENCE = "reference"  # worth keeping long term
    USEFUL = "useful"
    EPHEMERAL = "ephemeral"  # codes, delivery updates, already-used tickets


# Caps applied both to the JSON schema sent to the model (keeps small models
# from looping) and when normalising what comes back.
LIST_LIMITS = {"key_points": 6, "entities": 12, "tags": 8, "urls": 8, "actions": 5}

_NULLISH = {"", "null", "none", "n/a", "unknown", "not visible", "-"}


def _enum_or(enum: type[Enum], value: Any, default: Enum, synonyms: dict | None = None) -> Enum:
    if isinstance(value, enum):
        return value
    text = str(value or "").strip().lower().replace(" ", "_").replace("-", "_")
    for member in enum:
        if member.value == text:
            return member
    if synonyms and text in synonyms:
        return synonyms[text]
    return default


def _clean(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _optional(text: Any) -> str | None:
    value = _clean(text)
    return None if value.lower() in _NULLISH else value


def slugify_tag(tag: str) -> str:
    """Obsidian-safe tag: lowercase, kebab-case, unicode letters kept."""
    tag = unicodedata.normalize("NFC", str(tag)).strip().lower().lstrip("#")
    tag = re.sub(r"[\s_]+", "-", tag)
    tag = re.sub(r"[^\w\-/]", "", tag)
    tag = re.sub(r"-{2,}", "-", tag).strip("-/")
    return "" if not tag or tag.isdigit() else tag


class Entity(BaseModel):
    name: str
    type: EntityType = EntityType.OTHER

    @field_validator("name", mode="before")
    @classmethod
    def _name(cls, value: Any) -> str:
        return _clean(value).strip("[]#")

    @field_validator("type", mode="before")
    @classmethod
    def _type(cls, value: Any) -> EntityType:
        return _enum_or(EntityType, value, EntityType.OTHER)


class Action(BaseModel):
    task: str
    due: str | None = None

    @field_validator("task", mode="before")
    @classmethod
    def _task(cls, value: Any) -> str:
        return _clean(value)

    @field_validator("due", mode="before")
    @classmethod
    def _due(cls, value: Any) -> str | None:
        text = _clean(value)
        return text if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text) else None


class Extraction(BaseModel):
    """Field order matters: the model writes the description first, which
    grounds the title and summary that follow."""

    description: str = ""
    kind: Kind = Kind.OTHER
    source_app: str | None = None
    title: str = ""
    summary: str = ""
    key_points: list[str] = []
    why_saved: str = ""
    entities: list[Entity] = []
    tags: list[str] = []
    urls: list[str] = []
    actions: list[Action] = []
    code: str | None = None
    code_language: str | None = None
    language: str | None = None
    keep_value: KeepValue = KeepValue.USEFUL
    sensitive: bool = False
    sensitive_reason: str | None = None

    @field_validator("description", "summary", "why_saved", mode="before")
    @classmethod
    def _text(cls, value: Any) -> str:
        return _clean(value)

    @field_validator("title", mode="before")
    @classmethod
    def _title(cls, value: Any) -> str:
        title = _clean(value).strip("\"'“”` ")
        title = re.sub(r"^(a\s+)?screenshot\s+(of|showing)\s+", "", title, flags=re.I)
        return title[:100].rstrip(" .")

    @field_validator("kind", mode="before")
    @classmethod
    def _kind(cls, value: Any) -> Kind:
        return _enum_or(Kind, value, Kind.OTHER, _KIND_SYNONYMS)

    @field_validator("keep_value", mode="before")
    @classmethod
    def _keep(cls, value: Any) -> KeepValue:
        return _enum_or(KeepValue, value, KeepValue.USEFUL)

    @field_validator("source_app", "code_language", "language", "sensitive_reason", mode="before")
    @classmethod
    def _opt(cls, value: Any) -> str | None:
        return _optional(value)

    @field_validator("code", mode="before")
    @classmethod
    def _code(cls, value: Any) -> str | None:
        text = str(value or "").strip("\n")
        text = re.sub(r"^```[\w+-]*\n?|\n?```$", "", text)
        return text if text.strip() and text.strip().lower() not in _NULLISH else None

    @field_validator("key_points", mode="before")
    @classmethod
    def _points(cls, value: Any) -> list[str]:
        points = [_clean(p).lstrip("-•* ") for p in (value or [])]
        return _dedupe([p for p in points if p])[: LIST_LIMITS["key_points"]]

    @field_validator("tags", mode="before")
    @classmethod
    def _tags(cls, value: Any) -> list[str]:
        if isinstance(value, str):
            value = re.split(r"[,;]+", value)
        tags = [slugify_tag(t) for t in (value or [])]
        tags = [t for t in tags if t and t not in {"screenshot", "screenshots", "image"}]
        return _dedupe(tags)[: LIST_LIMITS["tags"]]

    @field_validator("urls", mode="before")
    @classmethod
    def _urls(cls, value: Any) -> list[str]:
        urls = [_clean(u).rstrip(".,;)") for u in (value or [])]
        return _dedupe([u for u in urls if "." in u and " " not in u])[: LIST_LIMITS["urls"]]

    @field_validator("entities", mode="before")
    @classmethod
    def _entities(cls, value: Any) -> list[Any]:
        seen: set[str] = set()
        out = []
        for item in value or []:
            if isinstance(item, str):
                item = {"name": item}
            name = _clean(item.get("name") if isinstance(item, dict) else getattr(item, "name", ""))
            if name and name.lower() not in seen and name.lower() not in _NULLISH:
                seen.add(name.lower())
                out.append(item)
        return out[: LIST_LIMITS["entities"]]

    @field_validator("actions", mode="before")
    @classmethod
    def _actions(cls, value: Any) -> list[Any]:
        out = []
        for item in value or []:
            if isinstance(item, str):
                item = {"task": item}
            if _clean(item.get("task") if isinstance(item, dict) else getattr(item, "task", "")):
                out.append(item)
        return out[: LIST_LIMITS["actions"]]


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out = []
    for item in items:
        if item.lower() not in seen:
            seen.add(item.lower())
            out.append(item)
    return out


def wire_schema() -> dict[str, Any]:
    """JSON schema handed to Ollama `format` / OpenAI `response_format`.

    Refs are inlined and every property is required so that grammar-constrained
    decoding works the same on llama.cpp, MLX and Ollama backends.
    """
    raw = Extraction.model_json_schema()
    defs = raw.pop("$defs", {})

    def resolve(node: Any) -> Any:
        if isinstance(node, list):
            return [resolve(n) for n in node]
        if not isinstance(node, dict):
            return node
        if "$ref" in node:
            target = copy.deepcopy(defs[node["$ref"].split("/")[-1]])
            extra = {k: v for k, v in node.items() if k != "$ref"}
            return resolve({**target, **extra})
        out: dict[str, Any] = {}
        for key, value in node.items():
            if key in ("title", "default", "description"):
                continue  # schema annotations, not constraints
            if key == "properties":
                out[key] = {name: resolve(sub) for name, sub in value.items()}
            else:
                out[key] = resolve(value)
        if out.get("type") == "object" and "properties" in out:
            out["required"] = list(out["properties"])
            out["additionalProperties"] = False
        return out

    schema = resolve(raw)
    for name, limit in LIST_LIMITS.items():
        schema["properties"][name]["maxItems"] = limit
    return schema
