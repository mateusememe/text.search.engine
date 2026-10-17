"""SQLite state: every screenshot, its OCR, extraction, embedding and search index.

The vault is a *rendering* of this database, so notes can be regenerated
(new template, better model) without re-running anything expensive.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np

from shotvault.models import Extraction

SCHEMA = """
CREATE TABLE IF NOT EXISTS shots (
    id            TEXT PRIMARY KEY,
    sha256        TEXT NOT NULL UNIQUE,
    phash         TEXT,
    source        TEXT NOT NULL,
    source_ref    TEXT,
    original_name TEXT,
    ext           TEXT,
    taken_at      TEXT,
    width         INTEGER,
    height        INTEGER,
    meta          TEXT NOT NULL DEFAULT '{}',
    status        TEXT NOT NULL DEFAULT 'new',
    duplicate_of  TEXT,
    ocr_engine    TEXT,
    ocr_text      TEXT,
    extraction    TEXT,
    extractor     TEXT,
    extracted_at  TEXT,
    error         TEXT,
    attempts      INTEGER NOT NULL DEFAULT 0,
    note_path     TEXT,
    rendered_tags TEXT,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS shots_status ON shots(status);
CREATE INDEX IF NOT EXISTS shots_taken ON shots(taken_at);

CREATE TABLE IF NOT EXISTS seen (
    source  TEXT NOT NULL,
    ref     TEXT NOT NULL,
    shot_id TEXT,
    seen_at TEXT NOT NULL,
    PRIMARY KEY (source, ref)
);

CREATE TABLE IF NOT EXISTS embeddings (
    shot_id   TEXT PRIMARY KEY,
    model     TEXT NOT NULL,
    dim       INTEGER NOT NULL,
    vector    BLOB NOT NULL,
    text_hash TEXT
);

CREATE TABLE IF NOT EXISTS hubs (
    name         TEXT PRIMARY KEY,
    members_hash TEXT,
    body         TEXT,
    model        TEXT,
    updated_at   TEXT
);

CREATE VIRTUAL TABLE IF NOT EXISTS shots_fts USING fts5(
    shot_id UNINDEXED, title, summary, body, tags, entities,
    tokenize = 'unicode61 remove_diacritics 2'
);
"""

# Statuses: new -> ocr_done -> extracted; side exits: duplicate, error.
NEW, OCR_DONE, EXTRACTED, DUPLICATE, ERROR = "new", "ocr_done", "extracted", "duplicate", "error"


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


@dataclass
class Shot:
    id: str
    sha256: str
    phash: str | None
    source: str
    source_ref: str | None
    original_name: str | None
    ext: str | None
    taken_at: str | None
    width: int | None
    height: int | None
    meta: str
    status: str
    duplicate_of: str | None
    ocr_engine: str | None
    ocr_text: str | None
    extraction: str | None
    extractor: str | None
    extracted_at: str | None
    error: str | None
    attempts: int
    note_path: str | None
    rendered_tags: str | None
    created_at: str
    updated_at: str

    @property
    def taken(self) -> datetime:
        return datetime.fromisoformat(self.taken_at or self.created_at)

    @property
    def meta_dict(self) -> dict[str, Any]:
        return json.loads(self.meta or "{}")

    @property
    def data(self) -> Extraction | None:
        return Extraction.model_validate_json(self.extraction) if self.extraction else None

    def original_path(self, originals_dir: Path) -> Path:
        return originals_dir / self.sha256[:2] / f"{self.sha256}{self.ext or '.png'}"


class Database:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.executescript(SCHEMA)
        self._vectors: tuple[str, list[str], np.ndarray] | None = None

    def close(self) -> None:
        self.conn.close()

    # -- seen sources -------------------------------------------------------
    def seen(self, source: str, ref: str) -> bool:
        row = self.conn.execute("SELECT 1 FROM seen WHERE source=? AND ref=?", (source, ref)).fetchone()
        return row is not None

    def mark_seen(self, source: str, ref: str, shot_id: str | None) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO seen(source, ref, shot_id, seen_at) VALUES (?,?,?,?)",
                (source, ref, shot_id, now()),
            )

    def refs_for(self, source: str, shot_ids: Iterable[str]) -> dict[str, list[str]]:
        ids = list(shot_ids)
        out: dict[str, list[str]] = {}
        for chunk in range(0, len(ids), 500):
            part = ids[chunk : chunk + 500]
            marks = ",".join("?" * len(part))
            rows = self.conn.execute(
                f"SELECT shot_id, ref FROM seen WHERE source=? AND shot_id IN ({marks}) ORDER BY seen_at",
                (source, *part),
            )
            for row in rows:
                out.setdefault(row["shot_id"], []).append(row["ref"])
        return out

    # -- shots --------------------------------------------------------------
    def add_shot(self, **fields: Any) -> Shot:
        fields.setdefault("meta", "{}")
        fields["created_at"] = fields["updated_at"] = now()
        cols = ",".join(fields)
        with self.conn:
            self.conn.execute(
                f"INSERT INTO shots({cols}) VALUES ({','.join('?' * len(fields))})", tuple(fields.values())
            )
        return self.get(fields["id"])  # type: ignore[return-value]

    def update(self, shot_id: str, **fields: Any) -> None:
        fields["updated_at"] = now()
        assignments = ",".join(f"{k}=?" for k in fields)
        with self.conn:
            self.conn.execute(f"UPDATE shots SET {assignments} WHERE id=?", (*fields.values(), shot_id))

    def get(self, shot_id: str) -> Shot | None:
        row = self.conn.execute("SELECT * FROM shots WHERE id=?", (shot_id,)).fetchone()
        return Shot(**row) if row else None

    def find(self, prefix: str) -> Shot | None:
        row = self.conn.execute("SELECT * FROM shots WHERE id LIKE ? LIMIT 1", (prefix + "%",)).fetchone()
        return Shot(**row) if row else None

    def by_sha(self, sha: str) -> Shot | None:
        row = self.conn.execute("SELECT * FROM shots WHERE sha256=?", (sha,)).fetchone()
        return Shot(**row) if row else None

    def shots(self, *statuses: str, newest_first: bool = True, limit: int | None = None) -> list[Shot]:
        sql = "SELECT * FROM shots"
        params: list[Any] = []
        if statuses:
            sql += f" WHERE status IN ({','.join('?' * len(statuses))})"
            params.extend(statuses)
        sql += " ORDER BY COALESCE(taken_at, created_at) " + ("DESC" if newest_first else "ASC")
        if limit:
            sql += " LIMIT ?"
            params.append(limit)
        return [Shot(**row) for row in self.conn.execute(sql, params)]

    def counts(self) -> dict[str, int]:
        rows = self.conn.execute("SELECT status, COUNT(*) AS n FROM shots GROUP BY status")
        return {row["status"]: row["n"] for row in rows}

    # -- full text ----------------------------------------------------------
    def index_text(self, shot: Shot) -> None:
        data = shot.data
        if data is None:
            return
        body = "\n".join(
            [data.description, data.why_saved, *data.key_points, shot.ocr_text or "", data.code or "", *data.urls]
        )
        with self.conn:
            self.conn.execute("DELETE FROM shots_fts WHERE shot_id=?", (shot.id,))
            self.conn.execute(
                "INSERT INTO shots_fts(shot_id, title, summary, body, tags, entities) VALUES (?,?,?,?,?,?)",
                (
                    shot.id,
                    data.title,
                    data.summary,
                    body,
                    " ".join(t.replace("-", " ") for t in data.tags),
                    " ".join(e.name for e in data.entities),
                ),
            )

    def search_text(self, match: str, limit: int = 50) -> list[str]:
        try:
            rows = self.conn.execute(
                "SELECT shot_id FROM shots_fts WHERE shots_fts MATCH ? "
                "ORDER BY bm25(shots_fts, 0, 6.0, 3.0, 1.0, 4.0, 4.0) LIMIT ?",
                (match, limit),
            )
            return [row["shot_id"] for row in rows]
        except sqlite3.OperationalError:
            return []

    # -- vectors ------------------------------------------------------------
    def set_embedding(self, shot_id: str, model: str, vector: list[float], text_hash: str) -> None:
        array = np.asarray(vector, dtype=np.float32)
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO embeddings(shot_id, model, dim, vector, text_hash) VALUES (?,?,?,?,?)",
                (shot_id, model, array.size, array.tobytes(), text_hash),
            )
        self._vectors = None

    def embedding_hashes(self, model: str) -> dict[str, str]:
        rows = self.conn.execute("SELECT shot_id, text_hash FROM embeddings WHERE model=?", (model,))
        return {row["shot_id"]: row["text_hash"] for row in rows}

    def vectors(self, model: str) -> tuple[list[str], np.ndarray]:
        """All embeddings for `model`, L2-normalised; brute-force cosine is plenty
        fast for tens of thousands of screenshots."""
        if self._vectors and self._vectors[0] == model:
            return self._vectors[1], self._vectors[2]
        ids, rows = [], []
        for row in self.conn.execute("SELECT shot_id, vector FROM embeddings WHERE model=?", (model,)):
            ids.append(row["shot_id"])
            rows.append(np.frombuffer(row["vector"], dtype=np.float32))
        matrix = np.vstack(rows) if rows else np.zeros((0, 1), dtype=np.float32)
        if rows:
            matrix = matrix / np.clip(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-9, None)
        self._vectors = (model, ids, matrix)
        return ids, matrix

    # -- hub cache ----------------------------------------------------------
    def hub(self, name: str) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM hubs WHERE name=?", (name,)).fetchone()

    def set_hub(self, name: str, members_hash: str, body: str, model: str) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO hubs(name, members_hash, body, model, updated_at) VALUES (?,?,?,?,?)",
                (name, members_hash, body, model, now()),
            )

    def iter_all(self) -> Iterator[Shot]:
        for row in self.conn.execute("SELECT * FROM shots"):
            yield Shot(**row)
