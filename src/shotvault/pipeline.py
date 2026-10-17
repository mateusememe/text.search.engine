"""ingest -> OCR (+ near-duplicate check) -> vision extraction -> embeddings.

Every stage is incremental and commits per screenshot, so a run can be
interrupted (lid closed, Ctrl-C) and resumed without losing work.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import re
from collections import Counter
from contextlib import contextmanager
from typing import Iterator

from pydantic import ValidationError
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.progress import TimeRemainingColumn

from shotvault.config import Config
from shotvault.db import DUPLICATE, ERROR, EXTRACTED, NEW, OCR_DONE, Database, Shot, now
from shotvault.imaging import device_hint, dhash, for_model, hamming, open_image, sha256_bytes, taken_at
from shotvault.llm import LLM, LLMError, LLMUnavailable, make_llm, parse_json
from shotvault.models import Extraction, wire_schema
from shotvault.ocr import OCREngine, make_ocr
from shotvault.prompts import EXTRACT_SYSTEM, EXTRACT_USER, PROMPT_VERSION
from shotvault.sources import Candidate, SourceUnavailable, clear_staging, make_sources

SOURCE_LABELS = {"photos": "iCloud Photos", "iphone": "iPhone (USB)", "folder": "folder"}
MAX_ATTEMPTS = 3


def doc_text(shot: Shot) -> str:
    """What gets embedded for semantic search."""
    data = shot.data
    if data is None:
        return ""
    parts = [
        data.title,
        data.summary,
        *data.key_points,
        data.why_saved,
        " ".join(data.tags),
        ", ".join(e.name for e in data.entities),
        data.source_app or "",
        (shot.ocr_text or "")[:1000],
    ]
    return "\n".join(p for p in parts if p)


def embed_prefix(model: str, query: bool) -> str:
    if "nomic" in model:
        return "search_query: " if query else "search_document: "
    return ""


class Pipeline:
    def __init__(
        self,
        cfg: Config,
        db: Database,
        console: Console | None = None,
        llm: LLM | None = None,
        ocr: OCREngine | None = None,
    ):
        self.cfg = cfg
        self.db = db
        self.console = console or Console(stderr=True)
        self._llm = llm
        self._ocr = ocr

    @property
    def llm(self) -> LLM:
        if self._llm is None:
            self._llm = make_llm(self.cfg.llm)
        return self._llm

    @property
    def ocr(self) -> OCREngine:
        if self._ocr is None:
            self._ocr = make_ocr(self.cfg.ocr)
        return self._ocr

    @property
    def extractor_id(self) -> str:
        return f"{self.cfg.llm.vision_model}|prompt-v{PROMPT_VERSION}"

    @contextmanager
    def _progress(self) -> Iterator[Progress]:
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TextColumn("eta"),
            TimeRemainingColumn(),
            TextColumn("{task.fields[note]}", style="dim"),
            console=self.console,
        )
        with progress:
            yield progress

    # -- ingest -------------------------------------------------------------
    def ingest(self, only: list[str] | None = None) -> Counter:
        stats: Counter = Counter()
        self.cfg.originals_dir.mkdir(parents=True, exist_ok=True)
        for source in make_sources(self.cfg, only):
            label = SOURCE_LABELS.get(source.name, source.name)
            with self.console.status(f"Scanning {label}…"):
                try:
                    for candidate in source.scan(self.db.seen):
                        self._ingest_one(candidate, stats)
                except SourceUnavailable as exc:
                    style = "yellow" if only else "dim"
                    self.console.print(f"[{style}]· {label}: skipped ({exc})")
                except Exception as exc:  # permissions, device unplugged mid-copy...
                    self.console.print(f"[red]· {label}: {type(exc).__name__}: {exc}")
                    stats["source_errors"] += 1
        clear_staging(self.cfg)
        return stats

    def _ingest_one(self, candidate: Candidate, stats: Counter) -> Shot | None:
        try:
            if candidate.rejected or candidate.path is None:
                self.db.mark_seen(candidate.source, candidate.ref, None)
                stats["not_screenshots"] += 1
                return None
            data = candidate.path.read_bytes()
            sha = sha256_bytes(data)
            existing = self.db.by_sha(sha)
            if existing:
                self._merge_meta(existing, candidate)
                self.db.mark_seen(candidate.source, candidate.ref, existing.id)
                stats["already_known"] += 1
                return existing
            try:
                image = open_image(data)
            except Exception:
                self.db.mark_seen(candidate.source, candidate.ref, None)
                stats["unreadable"] += 1
                return None
            ext = candidate.path.suffix.lower() or ".png"
            dest = self.cfg.originals_dir / sha[:2] / f"{sha}{ext}"
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                dest.write_bytes(data)
            when = (
                candidate.taken_at
                if candidate.source == "photos"
                else taken_at(data, candidate.name, candidate.taken_at)
            )
            shot = self.db.add_shot(
                id=sha[:12],
                sha256=sha,
                phash=dhash(image),
                source=candidate.source,
                source_ref=candidate.ref,
                original_name=candidate.name,
                ext=ext,
                taken_at=when.isoformat(timespec="seconds") if when else None,
                width=image.width,
                height=image.height,
                meta=json.dumps(candidate.meta, ensure_ascii=False),
                status=NEW,
            )
            self.db.mark_seen(candidate.source, candidate.ref, shot.id)
            stats["new"] += 1
            return shot
        finally:
            if candidate.temporary and candidate.path:
                candidate.path.unlink(missing_ok=True)

    def _merge_meta(self, shot: Shot, candidate: Candidate) -> None:
        """Same bytes seen via another source (e.g. USB first, iCloud later): keep the richer metadata."""
        meta = shot.meta_dict
        added = {k: v for k, v in candidate.meta.items() if k not in meta}
        if added:
            self.db.update(shot.id, meta=json.dumps({**meta, **added}, ensure_ascii=False))

    # -- OCR + near duplicates ---------------------------------------------
    def run_ocr(self, limit: int | None = None) -> Counter:
        stats: Counter = Counter()
        pending = self.db.shots(NEW, limit=limit)
        if not pending:
            return stats
        known = [(s.id, s.phash, s.ocr_text or "") for s in self.db.shots(OCR_DONE, EXTRACTED, ERROR)]
        with self._progress() as progress:
            task = progress.add_task(f"OCR ({self.ocr.name})", total=len(pending), note="")
            for shot in pending:
                path = shot.original_path(self.cfg.originals_dir)
                try:
                    text = self.ocr.recognize(path)
                except Exception as exc:
                    text = ""
                    stats["ocr_failed"] += 1
                    progress.console.print(f"[yellow]OCR failed for {shot.id}: {exc}")
                duplicate = self._find_duplicate(shot, text, known)
                if duplicate:
                    self.db.update(
                        shot.id, ocr_text=text, ocr_engine=self.ocr.name, status=DUPLICATE, duplicate_of=duplicate
                    )
                    stats["duplicates"] += 1
                else:
                    self.db.update(shot.id, ocr_text=text, ocr_engine=self.ocr.name, status=OCR_DONE)
                    known.append((shot.id, shot.phash, text))
                    stats["ocr"] += 1
                progress.advance(task)
        return stats

    @staticmethod
    def _find_duplicate(shot: Shot, text: str, known: list[tuple[str, str | None, str]]) -> str | None:
        """Perceptual hash finds look-alikes; OCR text confirms they say the same
        thing (two different chats in the same app look alike at 16x16)."""
        if not shot.phash:
            return None
        for other_id, other_hash, other_text in known:
            if not other_hash:
                continue
            distance = hamming(shot.phash, other_hash)
            if text.strip() and other_text.strip():
                if distance <= 24:
                    matcher = difflib.SequenceMatcher(None, text[:4000], other_text[:4000], autojunk=False)
                    if matcher.quick_ratio() >= 0.92 and matcher.ratio() >= 0.92:
                        return other_id
            elif not text.strip() and not other_text.strip() and distance <= 6:
                return other_id
        return None

    # -- vision extraction ---------------------------------------------------
    def run_extract(self, limit: int | None = None) -> Counter:
        stats: Counter = Counter()
        pending = self.db.shots(OCR_DONE, limit=limit)
        if not pending:
            return stats
        schema = wire_schema()
        system = EXTRACT_SYSTEM.format(language=self.cfg.language)
        with self._progress() as progress:
            task = progress.add_task(f"Understanding ({self.cfg.llm.vision_model})", total=len(pending), note="")
            for shot in pending:
                try:
                    data = self.extract_one(shot, schema, system)
                except LLMUnavailable:
                    raise
                except (LLMError, ValidationError, ValueError, OSError) as exc:
                    attempts = shot.attempts + 1
                    self.db.update(
                        shot.id,
                        attempts=attempts,
                        error=f"{type(exc).__name__}: {exc}"[:800],
                        status=ERROR if attempts >= MAX_ATTEMPTS else OCR_DONE,
                    )
                    stats["failed"] += 1
                    progress.console.print(f"[yellow]{shot.id}: {type(exc).__name__}: {str(exc)[:160]}")
                    progress.advance(task)
                    continue
                self.db.update(
                    shot.id,
                    extraction=data.model_dump_json(),
                    extractor=self.extractor_id,
                    extracted_at=now(),
                    status=EXTRACTED,
                    error=None,
                )
                self.db.index_text(self.db.get(shot.id))  # type: ignore[arg-type]
                stats["extracted"] += 1
                stats["sensitive"] += int(data.sensitive)
                progress.update(task, advance=1, note=data.title[:48])
        return stats

    def extract_one(self, shot: Shot, schema: dict | None = None, system: str | None = None) -> Extraction:
        image = for_model(
            shot.original_path(self.cfg.originals_dir), self.cfg.llm.max_image_pixels, self.cfg.llm.max_image_side
        )
        prompt = EXTRACT_USER.format(
            metadata=self.metadata(shot),
            ocr=(shot.ocr_text or "").strip()[: self.cfg.ocr.max_chars] or "(no text detected)",
        )
        raw = self.llm.vision_json(
            system or EXTRACT_SYSTEM.format(language=self.cfg.language), prompt, image, schema or wire_schema()
        )
        return ground(Extraction.model_validate(parse_json(raw)), shot)

    def metadata(self, shot: Shot) -> str:
        meta = shot.meta_dict
        lines = [
            f"- captured: {shot.taken.strftime('%A, %d %B %Y %H:%M')}",
            f"- device: {device_hint(shot.width or 0, shot.height or 0)} ({shot.width}x{shot.height})",
            f"- imported from: {SOURCE_LABELS.get(shot.source, shot.source)}",
        ]
        if meta.get("albums"):
            lines.append(f"- albums: {', '.join(meta['albums'])}")
        if meta.get("place"):
            lines.append(f"- place: {meta['place']}")
        for key in ("title", "description"):
            if meta.get(key):
                lines.append(f"- owner's {key}: {meta[key]}")
        if meta.get("keywords"):
            lines.append(f"- keywords: {', '.join(meta['keywords'])}")
        if shot.original_name:
            lines.append(f"- file: {shot.original_name}")
        return "\n".join(lines)

    # -- embeddings ---------------------------------------------------------
    def run_embed(self, batch: int = 16) -> Counter:
        stats: Counter = Counter()
        model = self.cfg.llm.embed_model
        if not model:
            return stats
        known = self.db.embedding_hashes(model)
        todo = []
        for shot in self.db.shots(EXTRACTED):
            text = doc_text(shot)
            digest = hashlib.sha1(text.encode()).hexdigest()
            if text and known.get(shot.id) != digest:
                todo.append((shot.id, text, digest))
        if not todo:
            return stats
        prefix = embed_prefix(model, query=False)
        with self._progress() as progress:
            task = progress.add_task(f"Embedding ({model})", total=len(todo), note="")
            for start in range(0, len(todo), batch):
                chunk = todo[start : start + batch]
                vectors = self.llm.embed([prefix + text for _, text, _ in chunk])
                if len(vectors) != len(chunk):
                    raise LLMError(f"asked for {len(chunk)} embeddings, got {len(vectors)}")
                for (shot_id, _, digest), vector in zip(chunk, vectors, strict=True):
                    self.db.set_embedding(shot_id, model, vector, digest)
                stats["embedded"] += len(chunk)
                progress.advance(task, len(chunk))
        return stats

    # -- maintenance --------------------------------------------------------
    def reset(self, shots: list[Shot], redo_ocr: bool = False) -> int:
        for shot in shots:
            if shot.status == DUPLICATE and not redo_ocr:
                continue
            self.db.update(shot.id, status=NEW if redo_ocr else OCR_DONE, attempts=0, error=None, duplicate_of=None)
        return len(shots)


def ground(data: Extraction, shot: Shot) -> Extraction:
    """Cheap anti-hallucination checks against the OCR text."""
    ocr = (shot.ocr_text or "").lower()
    if ocr:
        squashed = re.sub(r"\s+", "", ocr)

        def visible(url: str) -> bool:
            core = re.sub(r"^(https?://)?(www\.)?", "", url.lower()).rstrip("/")
            host = core.split("/")[0]
            return bool(host) and (host in squashed or core[:40] in squashed)

        data.urls = [u for u in data.urls if visible(u)]
    if not data.title:
        first_line = next((line.strip() for line in (shot.ocr_text or "").splitlines() if len(line.strip()) > 8), "")
        data.title = first_line[:70] or f"{data.kind.value.replace('_', ' ').title()} {shot.taken:%Y-%m-%d %H:%M}"
    return data
