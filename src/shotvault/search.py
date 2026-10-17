"""Hybrid search (SQLite FTS5 keywords + embedding similarity, fused with
reciprocal rank fusion) and retrieval-augmented answers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
from urllib.parse import quote

import numpy as np

from shotvault.config import Config
from shotvault.db import EXTRACTED, Database, Shot
from shotvault.llm import LLM, LLMError
from shotvault.models import Extraction
from shotvault.pipeline import embed_prefix
from shotvault.prompts import ASK_SYSTEM, ASK_USER


@dataclass
class Hit:
    shot: Shot
    data: Extraction
    score: float

    def as_dict(self, cfg: Config) -> dict:
        return {
            "id": self.shot.id,
            "title": self.data.title,
            "kind": self.data.kind.value,
            "captured": self.shot.taken.isoformat(timespec="minutes"),
            "app": self.data.source_app,
            "summary": self.data.summary,
            "key_points": self.data.key_points,
            "tags": self.data.tags,
            "urls": self.data.urls,
            "note": self.shot.note_path,
            "obsidian_url": obsidian_uri(cfg, self.shot.note_path),
            "score": round(self.score, 4),
        }


def fts_query(text: str) -> str:
    tokens = [t for t in re.findall(r"\w+", text.lower()) if len(t) > 1]
    return " OR ".join(f'"{t}"*' for t in tokens)


def obsidian_uri(cfg: Config, note_path: str | None) -> str | None:
    if not note_path:
        return None
    file = str(Path(note_path).with_suffix(""))
    return f"obsidian://open?vault={quote(cfg.vault.name)}&file={quote(file)}"


def search(
    db: Database,
    cfg: Config,
    query: str,
    llm: LLM | None = None,
    limit: int = 10,
    kind: str | None = None,
    include_sensitive: bool = False,
    warn=None,
) -> list[Hit]:
    rankings: list[list[str]] = []
    match = fts_query(query)
    if match:
        rankings.append(db.search_text(match, 100))
    model = cfg.llm.embed_model
    if llm is not None and model:
        ids, matrix = db.vectors(model)
        if ids:
            try:
                vector = np.asarray(llm.embed([embed_prefix(model, query=True) + query])[0], dtype=np.float32)
                vector /= max(float(np.linalg.norm(vector)), 1e-9)
                order = np.argsort(-(matrix @ vector))[:100]
                rankings.append([ids[i] for i in order])
            except LLMError as exc:
                if warn:
                    warn(f"semantic search unavailable, using keywords only ({exc})")

    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, shot_id in enumerate(ranking):
            scores[shot_id] = scores.get(shot_id, 0.0) + 1.0 / (60 + rank)

    hits: list[Hit] = []
    for shot_id, score in sorted(scores.items(), key=lambda kv: -kv[1]):
        shot = db.get(shot_id)
        if shot is None or shot.status != EXTRACTED or shot.data is None:
            continue
        data = shot.data
        if data.sensitive and not include_sensitive and cfg.privacy.sensitive != "keep":
            continue
        if kind and data.kind.value != kind:
            continue
        hits.append(Hit(shot, data, score))
        if len(hits) >= limit:
            break
    return hits


def answer(
    db: Database, cfg: Config, llm: LLM, question: str, k: int = 8, warn=None
) -> tuple[Iterator[str], list[Hit]]:
    hits = search(db, cfg, question, llm=llm, limit=k, warn=warn)
    blocks = []
    for n, hit in enumerate(hits, 1):
        data = hit.data
        lines = [
            f"[{n}] {data.title} ({hit.shot.taken:%Y-%m-%d}, {data.kind.value}, {data.source_app or 'unknown app'})"
        ]
        lines.append(data.summary)
        if data.key_points:
            lines.append("Key points: " + "; ".join(data.key_points))
        if data.code:
            lines.append("Code:\n" + data.code[:800])
        if hit.shot.ocr_text:
            lines.append("Text excerpt: " + hit.shot.ocr_text[:700].replace("\n", " "))
        blocks.append("\n".join(lines))
    if not blocks:
        return iter(["I couldn't find any screenshot about that."]), []
    stream = llm.chat(ASK_SYSTEM, ASK_USER.format(context="\n\n".join(blocks), question=question), stream=True)
    return stream, hits
