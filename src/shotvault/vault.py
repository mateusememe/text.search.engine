"""Render the database into Obsidian notes.

Ownership rules, so the vault stays yours:
- shotvault only rewrites the block between its start/end markers and the
  frontmatter keys it owns; anything you write below the end marker, and any
  extra properties or tags you add, survive every re-render.
- Notes are found by their `shotvault:` id, so you can rename or move them.
- If you delete the markers from a note, shotvault never touches it again.
- If you delete a note, it stays deleted (`shotvault render --restore` brings it back).
- Files are only written when their content changes (quiet git history).
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

import yaml
from rich.console import Console

from shotvault.config import Config
from shotvault.db import DUPLICATE, EXTRACTED, Database, Shot
from shotvault.imaging import write_attachment
from shotvault.llm import LLM, LLMError
from shotvault.models import KIND_LABELS, Extraction, Kind, KeepValue
from shotvault.prompts import PROMPT_VERSION, TOPIC_SYSTEM, TOPIC_USER

START = "<!-- shotvault:start · generated: edits inside this block are replaced on the next run; write below the end marker -->"
END = "<!-- shotvault:end -->"
_START_RE = re.compile(r"<!-- shotvault:start.*?-->", re.DOTALL)
_ID_RE = re.compile(r"^shotvault:\s*['\"]?([0-9a-f]{12})", re.MULTILINE)
_HUB_RE = re.compile(r"^shotvault_hub:\s*['\"]?([^\s'\"]+)", re.MULTILINE)
_TASK_RE = re.compile(r"^\s*- \[([^\]])\] .*\^(sv-[0-9a-f]{12}-\d+)\s*$", re.MULTILINE)

OWNED_KEYS = [
    "shotvault", "title", "kind", "app", "captured", "source", "place", "tags",
    "entities", "urls", "keep", "sensitive", "model",
]  # fmt: skip

HOME = "Screenshots Home"
ACTIONS = "Screenshot Actions"
CLEANUP = "Screenshot Cleanup"
ON_THIS_DAY = "Screenshots On This Day"
SOURCE_NAMES = {"photos": "iCloud Photos", "iphone": "iPhone (USB)", "folder": "folder"}


# -- small helpers -------------------------------------------------------------
def safe_filename(text: str, max_len: int = 80) -> str:
    text = re.sub(r"[\\/:*?\"<>|#^\[\]\x00-\x1f]", " ", text)
    text = re.sub(r"\s+", " ", text).strip().strip(".")
    return text[:max_len].rstrip(" .") or "Untitled"


def alias(text: str) -> str:
    return re.sub(r"[\[\]|]", "", text).strip() or "untitled"


def link(target: str, label: str | None = None) -> str:
    return f"[[{target}|{alias(label)}]]" if label and label != target else f"[[{target}]]"


def escape(text: str) -> str:
    """Neutralise things Obsidian would interpret inside free text."""
    text = text.replace("<!--", "&lt;!--").replace("[[", "[​[")
    return re.sub(r"(^|\s)#(?=[^\s#])", r"\1\\#", text)


def topic_label(tag: str) -> str:
    return " / ".join(part.replace("-", " ").title() for part in tag.split("/"))


def topic_note(tag: str) -> str:
    return safe_filename(f"{topic_label(tag).replace(' / ', ' - ')} (screenshots)")


def entity_note(name: str) -> str:
    return safe_filename(name, 100)


def fence(code: str, lang: str | None) -> list[str]:
    ticks = "`" * max(3, max((len(m) for m in re.findall(r"`+", code)), default=0) + 1)
    return [f"{ticks}{(lang or '').lower()}", code.rstrip(), ticks]


def quote(text: str) -> list[str]:
    lines = []
    for line in text.strip().splitlines():
        line = re.sub(r"^(\s*)(#+\s|>)", r"\1\\\2", escape(line))
        lines.append(f"> {line}" if line.strip() else ">")
    return lines


def split_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    if text.startswith("---\n"):
        end = text.find("\n---", 4)
        if end != -1:
            try:
                data = yaml.safe_load(text[4:end]) or {}
            except yaml.YAMLError:
                data = None
            if isinstance(data, dict):
                return data, text[end + 4 :].lstrip("\n")
    return {}, text


def compose(frontmatter: dict[str, Any], body: str) -> str:
    fm = yaml.safe_dump(frontmatter, sort_keys=False, allow_unicode=True, default_flow_style=False, width=10**6)
    return f"---\n{fm}---\n{body.rstrip()}\n"


def merge(existing: str | None, owned: dict[str, Any], block: str, previous_tags: Iterable[str] = ()) -> str | None:
    """New file text, or None when the note was taken over by hand (markers removed)."""
    if existing is None:
        return compose(owned, f"{block}\n\n## Notes\n\n")
    old_fm, body = split_frontmatter(existing)
    match = _START_RE.search(body)
    end = body.find(END, match.end()) if match else -1
    if not match or end == -1:
        return None
    if "tags" in owned:
        previous = set(previous_tags) | set(owned["tags"])
        old_tags = old_fm.get("tags") or []
        old_tags = old_tags if isinstance(old_tags, list) else [old_tags]
        owned = {**owned, "tags": owned["tags"] + [t for t in old_tags if t not in previous]}
    frontmatter = {**owned, **{k: v for k, v in old_fm.items() if k not in OWNED_KEYS and k not in owned}}
    return compose(frontmatter, body[: match.start()] + block + body[end + len(END) :])


def write_if_changed(path: Path, text: str) -> bool:
    if path.exists() and path.read_text(encoding="utf-8") == text:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
    return True


@dataclass
class VaultIndex:
    notes: dict[str, Path] = field(default_factory=dict)  # shot id -> note
    hubs: dict[str, Path] = field(default_factory=dict)  # hub key -> file
    names: set[str] = field(default_factory=set)  # lower-cased basenames of *your* notes


def scan_vault(vault: Path) -> VaultIndex:
    index = VaultIndex()
    if not vault.exists():
        return index
    for dirpath, dirnames, filenames in os.walk(vault):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for filename in filenames:
            if not filename.endswith(".md"):
                continue
            path = Path(dirpath) / filename
            try:
                with path.open(encoding="utf-8", errors="ignore") as handle:
                    head = handle.read(2048)
            except OSError:
                continue
            if head.startswith("---"):
                if match := _ID_RE.search(head):
                    index.notes[match.group(1)] = path
                    continue
                if match := _HUB_RE.search(head):
                    index.hubs[match.group(1)] = path
                    continue
            index.names.add(path.stem.lower())
    return index


@dataclass
class RenderStats:
    written: int = 0
    unchanged: int = 0
    hand_edited: int = 0
    deleted_by_you: int = 0
    hubs: int = 0
    topics_synthesized: int = 0


class VaultRenderer:
    def __init__(
        self,
        cfg: Config,
        db: Database,
        llm: LLM | None = None,
        console: Console | None = None,
        restore: bool = False,
    ):
        self.cfg = cfg
        self.restore = restore
        self.db = db
        self.llm = llm
        self.console = console or Console(stderr=True)
        self.root = cfg.root
        self.index = scan_vault(cfg.vault)
        self.stats = RenderStats()
        self.today = date.today()

    # -- public ---------------------------------------------------------------
    def render(self) -> RenderStats:
        shots = [s for s in self.db.shots(EXTRACTED) if s.data is not None]
        data = {s.id: s.data for s in shots}  # parse each extraction once
        mode = self.cfg.privacy.sensitive
        self.private = {s.id for s in shots if data[s.id].sensitive and mode != "keep"}  # type: ignore[union-attr]
        deleted = {s.id for s in shots if self._deleted(s)}
        self.stats.deleted_by_you = len(deleted)
        # skip mode writes no note, but a note that already exists gets redacted rather than left as is
        noted = [
            s
            for s in shots
            if s.id not in deleted and not (s.id in self.private and mode == "skip" and not self._existing(s))
        ]
        visible = [s for s in noted if s.id not in self.private]

        self.data = data
        self.by_id = {s.id: s for s in shots}
        self.names: dict[str, str] = {}  # shot id -> note basename
        self.titles: dict[str, str] = {}
        self.entities = self._canonical_entities(visible)
        self.topics = self._topics(visible)
        self.neighbours = self._neighbours(visible)

        planned = {s.id: self._note_path(s) for s in noted}
        for shot in noted:
            self.names[shot.id] = planned[shot.id].stem
            self.titles[shot.id] = self._title(shot)
        for shot in noted:
            self._write_note(shot, planned[shot.id])

        self._hub_actions(visible)
        self._hub_cleanup(shots)
        if self.cfg.knowledge.topics:
            self._hub_topics(visible)
        if self.cfg.knowledge.entities:
            self._hub_entities(visible)
        if self.cfg.knowledge.monthly:
            self._hub_months(noted)
        if self.cfg.knowledge.on_this_day:
            self._hub_on_this_day(visible)
        self._hub_home(shots, noted)
        return self.stats

    # -- notes ----------------------------------------------------------------
    def _title(self, shot: Shot) -> str:
        data = self.data[shot.id]
        if shot.id in self.private:
            return f"Private {KIND_LABELS[data.kind].split(' ', 1)[1].lower()} screenshot"
        return data.title or shot.original_name or shot.id

    def _existing(self, shot: Shot) -> Path | None:
        found = self.index.notes.get(shot.id)
        if found and found.exists():
            return found
        if shot.note_path and (self.cfg.vault / shot.note_path).exists():
            return self.cfg.vault / shot.note_path
        return None

    def _deleted(self, shot: Shot) -> bool:
        return bool(shot.note_path) and not self.restore and self._existing(shot) is None

    def _note_path(self, shot: Shot) -> Path:
        found = self._existing(shot)
        if found and shot.id in self.private and not found.stem.split(" ", 1)[-1].startswith("Private "):
            # flagged sensitive after it was written: drop the descriptive file name too
            renamed = self._new_path(shot)
            renamed.parent.mkdir(parents=True, exist_ok=True)
            os.replace(found, renamed)
            return renamed
        return found or self._new_path(shot)

    def _new_path(self, shot: Shot) -> Path:
        taken = shot.taken
        folder = self.root / f"{taken:%Y}" / f"{taken:%m}"
        stem = f"{taken:%Y-%m-%d} {safe_filename(self._title(shot), 70)}"
        if stem.lower() in self.index.names or (folder / f"{stem}.md").exists():
            stem = f"{stem} ({shot.id[:6]})"
        self.index.names.add(stem.lower())
        path = folder / f"{stem}.md"
        self.index.notes[shot.id] = path
        return path

    def _attachment(self, shot: Shot) -> str | None:
        cfg = self.cfg.attachments
        ext = {"jpeg": "jpg"}.get(cfg.format, cfg.format)
        dest = self.root / "_attachments" / f"{shot.taken:%Y}" / f"{shot.taken:%m}" / f"{shot.id}.{ext}"
        if shot.id in self.private or cfg.mode == "none":
            dest.unlink(missing_ok=True)  # e.g. re-classified as sensitive
            return None
        if not dest.exists():
            source = shot.original_path(self.cfg.originals_dir)
            if not source.exists():
                return None
            write_attachment(source, dest, cfg.format, cfg.quality, cfg.max_side)
        return dest.name

    def _write_note(self, shot: Shot, path: Path) -> None:
        data = self.data[shot.id]
        private = shot.id in self.private
        meta = shot.meta_dict
        tags = ["screenshot", f"screenshot/{data.kind.value.replace('_', '-')}"]
        tags += ["screenshot/private"] if private else data.tags
        owned: dict[str, Any] = {
            "shotvault": shot.id,
            "title": self.titles[shot.id],
            "kind": data.kind.value,
            "app": None if private else data.source_app,
            "captured": shot.taken.isoformat(timespec="minutes"),
            "source": SOURCE_NAMES.get(shot.source, shot.source),
            "place": None if private else meta.get("place"),
            "tags": tags,
            "entities": [] if private else [f"[[{self.entity_name(e.name)}]]" for e in data.entities],
            "urls": [] if private else data.urls,
            "keep": data.keep_value.value,
            "sensitive": bool(data.sensitive),
            "model": (shot.extractor or "").split("|")[0],
        }
        owned = {k: v for k, v in owned.items() if v not in (None, [], "")}
        if private:
            self._attachment(shot)  # removes an image copied before it was flagged
            block = self._private_block(shot, data)
        else:
            block = self._block(shot, data)
        previous = json.loads(shot.rendered_tags or "[]")
        existing = path.read_text(encoding="utf-8") if path.exists() else None
        text = merge(existing, owned, block, previous)
        if text is None:
            self.stats.hand_edited += 1
            return
        if write_if_changed(path, text):
            self.stats.written += 1
        else:
            self.stats.unchanged += 1
        relative = str(path.relative_to(self.cfg.vault)) if path.is_relative_to(self.cfg.vault) else str(path)
        if relative != shot.note_path or shot.rendered_tags != json.dumps(tags):
            self.db.update(shot.id, note_path=relative, rendered_tags=json.dumps(tags))

    def _block(self, shot: Shot, data: Extraction) -> str:
        meta = shot.meta_dict
        out = [START, f"# {escape(self.titles[shot.id])}", ""]
        if attachment := self._attachment(shot):
            out += [f"![[{attachment}|{self.cfg.attachments.embed_width}]]", ""]
        if data.summary:
            out += ["> [!summary]", *quote(data.summary), ""]
        facts = []
        if data.description:
            facts.append(f"**What it shows:** {escape(data.description)}")
        if data.why_saved:
            facts.append(f"**Why you saved it:** {escape(data.why_saved)}")
        context = [f"{shot.taken:%a %d %b %Y, %H:%M}", data.source_app, meta.get("place")]
        context.append(f"via {SOURCE_NAMES.get(shot.source, shot.source)}")
        facts.append(f"**Captured:** {' · '.join(escape(c) for c in context if c)}")
        out += ["  \n".join(facts), ""]
        if data.key_points:
            out += ["### Key points", *[f"- {escape(p)}" for p in data.key_points], ""]
        if data.code:
            out += ["### Code", *fence(data.code, data.code_language), ""]
        if data.actions:
            out += ["### To do"]
            for action in data.actions:
                due = f" (due {action.due})" if action.due else ""
                out.append(f"- {escape(action.task)}{due}")
            out += [f"- *Tick them off in [[{ACTIONS}]]*", ""]
        if data.urls:
            out += ["### Links", *[f"- <{u if '://' in u else 'https://' + u}>" for u in data.urls], ""]
        connections = []
        if data.entities:
            connections.append("**Mentions:** " + " · ".join(f"[[{self.entity_name(e.name)}]]" for e in data.entities))
        topics = [t for t in data.tags if t in self.topics]
        if topics:
            connections.append("**Topics:** " + " · ".join(link(topic_note(t), topic_label(t)) for t in topics))
        nearby = self.neighbours.get(shot.id, [])
        if nearby:
            connections.append(
                "**Same moment:** " + " · ".join(link(self.names[n], self.titles[n]) for n in nearby if n in self.names)
            )
        if connections:
            out += ["### Connections", "  \n".join(connections), ""]
        if (shot.ocr_text or "").strip():
            out += ["> [!quote]- Text in the screenshot", *quote(shot.ocr_text or ""), ""]
        out.append(END)
        return "\n".join(out)

    def _private_block(self, shot: Shot, data: Extraction) -> str:
        reason = data.sensitive_reason or "personal information"
        return "\n".join(
            [
                START,
                f"# {self.titles[shot.id]}",
                "",
                "> [!warning] Sensitive screenshot",
                f"> Flagged as sensitive ({escape(reason)}). The image and its text stay in the local",
                "> shotvault database and never enter the vault.",
                f"> Open it on this Mac with `shotvault show {shot.id}`.",
                "",
                f"**Captured:** {shot.taken:%a %d %b %Y, %H:%M} · via {SOURCE_NAMES.get(shot.source, shot.source)}",
                "",
                END,
            ]
        )

    # -- knowledge structure --------------------------------------------------
    def _canonical_entities(self, shots: list[Shot]) -> dict[str, dict[str, Any]]:
        spellings: dict[str, Counter] = defaultdict(Counter)
        types: dict[str, Counter] = defaultdict(Counter)
        mentions: dict[str, list[str]] = defaultdict(list)
        for shot in shots:
            for entity in self.data[shot.id].entities:
                key = entity.name.lower()
                spellings[key][entity.name] += 1
                types[key][entity.type.value] += 1
                mentions[key].append(shot.id)
        return {
            key: {
                # most frequent spelling; on ties prefer capitalised ("Rust" over "rust")
                "name": max(spellings[key].items(), key=lambda kv: (kv[1], kv[0] != kv[0].lower()))[0],
                "type": types[key].most_common(1)[0][0],
                "shots": ids,
            }
            for key, ids in mentions.items()
        }

    def entity_name(self, name: str) -> str:
        entry = getattr(self, "entities", {}).get(name.lower())
        return entity_note(entry["name"] if entry else name)

    def _topics(self, shots: list[Shot]) -> dict[str, list[str]]:
        members: dict[str, list[str]] = defaultdict(list)
        for shot in shots:
            for tag in self.data[shot.id].tags:
                members[tag].append(shot.id)
        eligible = [(t, ids) for t, ids in members.items() if len(ids) >= self.cfg.knowledge.topic_min_shots]
        eligible.sort(key=lambda item: -len(item[1]))
        return dict(eligible[: self.cfg.knowledge.max_topics]) if self.cfg.knowledge.topics else {}

    def _neighbours(self, shots: list[Shot], window: timedelta = timedelta(minutes=10)) -> dict[str, list[str]]:
        ordered = sorted(shots, key=lambda s: s.taken)
        result: dict[str, list[str]] = defaultdict(list)
        for i, shot in enumerate(ordered):
            j = i + 1
            while j < len(ordered) and ordered[j].taken - shot.taken <= window:
                result[shot.id].append(ordered[j].id)
                result[ordered[j].id].append(shot.id)
                j += 1
        return {k: v[:5] for k, v in result.items()}

    # -- hubs -----------------------------------------------------------------
    def _write_hub(self, key: str, path: Path, body_lines: list[str]) -> None:
        found = self.index.hubs.get(key)
        if found and found.exists():
            path = found
        block = "\n".join([START, *body_lines, END])
        existing = path.read_text(encoding="utf-8") if path.exists() else None
        if existing is None:
            text = compose({"shotvault_hub": key}, block + "\n")
        else:
            text = merge(existing, {"shotvault_hub": key}, block)
            if text is None:
                return
        if write_if_changed(path, text):
            self.stats.hubs += 1

    def _remove_hub(self, key: str) -> None:
        """Delete a generated hub, unless you've written in it."""
        path = self.index.hubs.get(key)
        if not path or not path.exists():
            return
        _, body = split_frontmatter(path.read_text(encoding="utf-8"))
        tail = body.split(END, 1)[1] if END in body else "x"
        if not tail.strip():
            path.unlink()

    def _item(self, shot_id: str, with_date: bool = True, extra: str = "") -> str:
        shot_date = self.by_id[shot_id].taken if with_date else None
        prefix = f"{shot_date:%Y-%m-%d} · " if shot_date else ""
        return f"- {prefix}{link(self.names[shot_id], self.titles[shot_id])}{extra}"

    def _hub_actions(self, shots: list[Shot]) -> None:
        path = self.root / f"{ACTIONS}.md"
        found = self.index.hubs.get("actions")
        existing = (found or path).read_text(encoding="utf-8") if (found or path).exists() else ""
        ticked = {block_id: mark for mark, block_id in _TASK_RE.findall(existing)}
        by_month: dict[str, list[str]] = defaultdict(list)
        open_count = done = 0
        for shot in sorted(shots, key=lambda s: s.taken, reverse=True):
            for i, action in enumerate(self.data[shot.id].actions):
                block_id = f"sv-{shot.id}-{i}"
                mark = ticked.get(block_id, " ")
                open_count += mark == " "
                done += mark != " "
                due = f" 📅 {action.due}" if action.due else ""
                by_month[f"{shot.taken:%Y-%m}"].append(
                    f"- [{mark}] {escape(action.task)}{due} — {link(self.names[shot.id], self.titles[shot.id])} ^{block_id}"
                )
        lines = [
            "# Screenshot actions",
            "",
            f"*{open_count} open · {done} done. Tick boxes right here; shotvault keeps your ticks.*",
            "",
        ]
        for month in sorted(by_month, reverse=True):
            lines += [f"## {month}", *by_month[month], ""]
        self._write_hub("actions", path, lines)

    def _hub_cleanup(self, shots: list[Shot]) -> None:
        ephemeral = [s for s in shots if self.data[s.id].keep_value == KeepValue.EPHEMERAL and s.id in self.names]
        duplicates = [s for s in self.db.shots(DUPLICATE) if s.duplicate_of in self.names]
        lines = [
            "# Screenshot cleanup",
            "",
            "These look temporary or are duplicates, and anything worth keeping is already in your vault.",
            "Run `shotvault cleanup-album` to collect them in a Photos album, then review and delete them on your iPhone.",
            "",
            f"## Temporary ({len(ephemeral)})",
            *[self._item(s.id, extra=f" · `{s.original_name}`") for s in ephemeral],
            "",
            f"## Duplicates ({len(duplicates)})",
        ]
        for dup in duplicates:
            original = dup.duplicate_of or ""
            lines.append(
                f"- {dup.taken:%Y-%m-%d %H:%M} · `{dup.original_name}` → same as "
                f"{link(self.names[original], self.titles[original])}"
            )
        self._write_hub("cleanup", self.root / f"{CLEANUP}.md", lines)

    def _hub_topics(self, shots: list[Shot]) -> None:
        by_id = {s.id: s for s in shots}
        synthesis_on = self.cfg.knowledge.topic_synthesis and self.llm is not None
        for tag, ids in self.topics.items():
            members = sorted((by_id[i] for i in ids), key=lambda s: s.taken, reverse=True)
            synthesis = ""
            if synthesis_on:
                try:
                    synthesis = self._synthesis(tag, members)
                except LLMError as exc:
                    self.console.print(f"[yellow]Topic synthesis paused: {exc}")
                    synthesis_on = False
            if not synthesis:
                cached = self.db.hub(f"topic:{tag}")
                synthesis = self._cited(json.loads(cached["body"])) if cached else ""
            lines = [f"# {topic_label(tag)}", "", f"*{len(members)} screenshots tagged #{tag}*", ""]
            if synthesis:
                lines += [synthesis, ""]
            lines += ["## Screenshots", *[self._item(s.id) for s in members], ""]
            self._write_hub(f"topic:{tag}", self.root / "Topics" / f"{topic_note(tag)}.md", lines)
        for key in list(self.index.hubs):
            if key.startswith("topic:") and key[6:] not in self.topics:
                self._remove_hub(key)

    def _synthesis(self, tag: str, members: list[Shot]) -> str:
        subset = members[: self.cfg.knowledge.topic_max_entries]
        fingerprint = "|".join(sorted(f"{s.id}:{s.extractor}" for s in subset))
        fingerprint += f"|{self.cfg.language}|{self.cfg.llm.chat_model}|{PROMPT_VERSION}"
        members_hash = hashlib.sha1(fingerprint.encode()).hexdigest()
        cached = self.db.hub(f"topic:{tag}")
        if cached and cached["members_hash"] == members_hash:
            return self._cited(json.loads(cached["body"]))
        entries = []
        for n, shot in enumerate(subset, 1):
            data = self.data[shot.id]
            points = "; ".join(data.key_points[:3])
            entries.append(f"[{n}] {shot.taken:%Y-%m-%d} — {data.title}: {data.summary} {points}".strip())
        subset_note = f" (the {len(subset)} most recent of {len(members)})" if len(members) > len(subset) else ""
        assert self.llm is not None
        with self.console.status(f"Synthesizing topic #{tag}…"):
            text = "".join(
                self.llm.chat(
                    TOPIC_SYSTEM.format(language=self.cfg.language),
                    TOPIC_USER.format(count=len(subset), tag=tag, subset=subset_note, entries="\n".join(entries)),
                )
            )
        text = re.sub(r"^```(?:markdown)?\s*|\s*```$", "", text.strip())
        text = re.sub(r"^# .*\n", "", text).strip()
        payload = {"text": text, "ids": [s.id for s in subset]}
        self.db.set_hub(f"topic:{tag}", members_hash, json.dumps(payload), self.cfg.llm.chat_model)
        self.stats.topics_synthesized += 1
        return self._cited(payload)

    def _cited(self, payload: dict[str, Any]) -> str:
        ids = payload.get("ids", [])

        def replace(match: re.Match) -> str:
            refs = []
            for number in re.findall(r"\d+", match.group(1)):
                n = int(number)
                if 1 <= n <= len(ids) and ids[n - 1] in self.names:
                    refs.append(link(self.names[ids[n - 1]], number))
                else:
                    refs.append(number)
            return "(" + ", ".join(refs) + ")"

        return re.sub(r"\[(\d+(?:\s*,\s*\d+)*)\]", replace, escape(payload.get("text", "")))

    def _hub_entities(self, shots: list[Shot]) -> None:
        keep = set()
        for entry in self.entities.values():
            name = entity_note(entry["name"])
            hub_key = f"entity:{name.lower()}"
            if len(entry["shots"]) < self.cfg.knowledge.entity_min_mentions:
                continue
            if name.lower() in self.index.names:
                continue  # you already have a note with this name: links resolve to yours
            keep.add(hub_key)
            ids = sorted(set(entry["shots"]), key=lambda i: self.by_id[i].taken, reverse=True)
            lines = [
                f"# {entry['name']}",
                "",
                f"*{entry['type'].title()} · mentioned in {len(ids)} screenshots*",
                "",
            ]
            for shot_id in ids:
                summary = self.data[shot_id].summary
                first = re.split(r"(?<=[.!?])\s", summary, maxsplit=1)[0] if summary else ""
                lines.append(self._item(shot_id, extra=f" — {escape(first)}" if first else ""))
            self._write_hub(hub_key, self.root / "Entities" / f"{name}.md", lines)
        for key in list(self.index.hubs):
            if key.startswith("entity:") and key not in keep:
                self._remove_hub(key)

    def _hub_months(self, shots: list[Shot]) -> None:
        months: dict[str, list[Shot]] = defaultdict(list)
        for shot in shots:
            months[f"{shot.taken:%Y-%m}"].append(shot)
        for month, members in months.items():
            first = datetime.strptime(month, "%Y-%m")
            lines = [f"# {first:%B %Y}", "", f"*{len(members)} screenshots*", ""]
            by_kind: dict[Kind, list[Shot]] = defaultdict(list)
            for shot in members:
                by_kind[self.data[shot.id].kind].append(shot)
            for kind, group in sorted(by_kind.items(), key=lambda kv: -len(kv[1])):
                lines.append(f"## {KIND_LABELS[kind]} ({len(group)})")
                for shot in sorted(group, key=lambda s: s.taken):
                    lines.append(f"- {shot.taken:%d %H:%M} · {link(self.names[shot.id], self.titles[shot.id])}")
                lines.append("")
            self._write_hub(f"month:{month}", self.root / "Months" / f"Screenshots {month}.md", lines)

    def _hub_on_this_day(self, shots: list[Shot]) -> None:
        today = self.today
        past = [
            s for s in shots if (s.taken.month, s.taken.day) == (today.month, today.day) and s.taken.year < today.year
        ]
        lines = [f"# On this day · {today:%d %B}", ""]
        if past:
            for year in sorted({s.taken.year for s in past}, reverse=True):
                lines += [
                    f"## {year} ({today.year - year} years ago)" if today.year - year > 1 else f"## {year} (a year ago)"
                ]
                lines += [self._item(s.id, with_date=False) for s in past if s.taken.year == year] + [""]
        else:
            lines += ["*Nothing from this day in earlier years.*", ""]
        past_ids = {s.id for s in past}
        pool = [s for s in shots if self.data[s.id].keep_value == KeepValue.REFERENCE and s.id not in past_ids]
        if pool and self.cfg.knowledge.rediscover:
            picks = random.Random(today.toordinal()).sample(pool, min(self.cfg.knowledge.rediscover, len(pool)))
            lines += ["## Rediscover", *[self._item(s.id) for s in picks], ""]
        self._write_hub("on-this-day", self.root / f"{ON_THIS_DAY}.md", lines)

    def _hub_home(self, shots: list[Shot], noted: list[Shot]) -> None:
        kinds = Counter(self.data[s.id].kind for s in shots)
        recent = sorted(noted, key=lambda s: s.taken, reverse=True)[:20]
        week = sum(1 for s in shots if s.taken.date() >= self.today - timedelta(days=7))
        counts = self.db.counts()
        lines = [
            "# Screenshots",
            "",
            f"*{len(shots)} screenshots understood · {week} in the last 7 days · "
            f"{len(self.topics)} topics · updated {datetime.now():%Y-%m-%d %H:%M}*",
            "",
            f"{link(ACTIONS, '✅ Actions')} · {link(CLEANUP, '🧹 Cleanup')} · {link(ON_THIS_DAY, '📆 On this day')}",
            "",
        ]
        pending = counts.get("new", 0) + counts.get("ocr_done", 0)
        if pending or counts.get("error"):
            lines += [f"> [!info] {pending} waiting to be processed · {counts.get('error', 0)} failed", ""]
        lines += ["## Recent", *[self._item(s.id, extra=f" · {KIND_LABELS[self.data[s.id].kind]}") for s in recent], ""]
        if self.topics:
            lines += [
                "## Topics",
                " · ".join(f"{link(topic_note(t), topic_label(t))} ({len(ids)})" for t, ids in self.topics.items()),
                "",
            ]
        lines += ["## By kind", "| Kind | Screenshots |", "| --- | ---: |"]
        lines += [f"| {KIND_LABELS[k]} | {n} |" for k, n in kinds.most_common()] + [""]
        months = Counter(f"{s.taken:%Y-%m}" for s in noted)
        if months and self.cfg.knowledge.monthly:
            lines += [
                "## Months",
                " · ".join(f"{link('Screenshots ' + m, m)} ({n})" for m, n in sorted(months.items(), reverse=True)),
                "",
            ]
        lines += [
            "## Dataview",
            "With the Dataview plugin, query the properties shotvault writes:",
            "```dataview",
            'TABLE kind, app, captured FROM #screenshot WHERE keep = "reference" SORT captured DESC LIMIT 25',
            "```",
        ]
        self._write_hub("home", self.root / f"{HOME}.md", lines)
