"""Command line interface: `shotvault --help`."""

from __future__ import annotations

import fcntl
import json
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional

import typer
from rich.console import Console
from rich.markup import escape as rich_escape
from rich.table import Table

from shotvault import __version__
from shotvault.config import (
    CONFIG_TEMPLATE,
    DEFAULT_CONFIG_PATH,
    Config,
    ConfigError,
    config_path,
    load_config,
)
from shotvault.db import DUPLICATE, ERROR, EXTRACTED, Database
from shotvault.llm import LLMError, LLMUnavailable, make_llm
from shotvault.models import KIND_LABELS, KeepValue
from shotvault.pipeline import Pipeline
from shotvault.vault import VaultRenderer, link, safe_filename

app = typer.Typer(
    help="Turn your screenshots into an Obsidian knowledge base with local models.",
    no_args_is_help=True,
    rich_markup_mode="rich",
    add_completion=False,
)
schedule_app = typer.Typer(help="Nightly run via launchd (macOS).", no_args_is_help=True)
app.add_typer(schedule_app, name="schedule")

console = Console(stderr=True)
out = Console()
_state: dict[str, Optional[Path]] = {"config": None}

PROVIDER_DEFAULTS = {
    "ollama": {"vision_model": "qwen2.5vl:7b", "embed_model": "bge-m3", "base_url": "http://localhost:11434"},
    "openai": {
        "vision_model": "qwen2.5-vl-7b-instruct",
        "embed_model": "text-embedding-nomic-embed-text-v1.5",
        "base_url": "http://localhost:1234/v1",
    },
}


@app.callback()
def main(
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Config file (default: ~/.config/shotvault/config.toml)."
    ),
    version: bool = typer.Option(False, "--version", help="Show the version."),
) -> None:
    if version:
        out.print(f"shotvault {__version__}")
        raise typer.Exit()
    _state["config"] = config


def _cfg() -> Config:
    try:
        return load_config(_state["config"])
    except ConfigError as exc:
        console.print(f"[red]{exc}")
        raise typer.Exit(1) from exc


@contextmanager
def _locked(cfg: Config, skip_if_busy: bool = False) -> Iterator[bool]:
    """One run at a time (the nightly job and a manual run must not overlap)."""
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    handle = (cfg.data_dir / "run.lock").open("w")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        if skip_if_busy:
            yield False
            return
        console.print("[yellow]Another shotvault run is in progress.")
        raise typer.Exit(1) from None
    try:
        yield True
    finally:
        fcntl.flock(handle, fcntl.LOCK_UN)
        handle.close()


def _fmt(stats: Counter) -> str:
    return ", ".join(f"{v} {k.replace('_', ' ')}" for k, v in stats.items() if v) or "nothing new"


# -- setup ---------------------------------------------------------------------
@app.command()
def init(
    vault: Path = typer.Argument(..., help="Path to your Obsidian vault."),
    provider: str = typer.Option("ollama", help="ollama or openai (LM Studio)."),
    vision_model: Optional[str] = typer.Option(None, help="Vision model name."),
    embed_model: Optional[str] = typer.Option(None, help="Embedding model name."),
    force: bool = typer.Option(False, help="Overwrite an existing config."),
) -> None:
    """Create the config file."""
    if provider not in PROVIDER_DEFAULTS:
        console.print("[red]provider must be 'ollama' or 'openai'")
        raise typer.Exit(1)
    target = (_state["config"] or DEFAULT_CONFIG_PATH).expanduser()
    if target.exists() and not force:
        console.print(f"[yellow]{target} already exists (use --force to overwrite).")
        raise typer.Exit(1)
    vault = vault.expanduser().resolve()
    if not vault.is_dir():
        console.print(f"[red]{vault} is not a folder.")
        raise typer.Exit(1)
    if not (vault / ".obsidian").exists():
        console.print(f"[yellow]{vault} has no .obsidian folder; is it an Obsidian vault?")
    defaults = PROVIDER_DEFAULTS[provider]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        CONFIG_TEMPLATE.format(
            vault=vault,
            provider=provider,
            base_url=defaults["base_url"],
            vision_model=vision_model or defaults["vision_model"],
            embed_model=embed_model or defaults["embed_model"],
        )
    )
    console.print(f"[green]Wrote {target}")
    if provider == "ollama":
        console.print(
            f"\nNext:\n  ollama pull {vision_model or defaults['vision_model']}\n"
            f"  ollama pull {embed_model or defaults['embed_model']}"
        )
    else:
        console.print("\nNext: load a vision model and an embedding model in LM Studio, then `lms server start`.")
    console.print("  shotvault doctor\n  shotvault run --limit 20   # try it on your 20 newest screenshots")


@app.command()
def doctor(smoke: bool = typer.Option(False, "--smoke", help="Also run one real extraction on a test image.")) -> None:
    """Check that everything shotvault needs is in place."""
    cfg = _cfg()
    ok = True

    def check(passed: bool, label: str, hint: str = "") -> None:
        nonlocal ok
        ok &= passed
        mark = "[green]✓" if passed else "[red]✗"
        console.print(f"{mark}[/] {label}" + (f" [dim]— {hint}" if hint and not passed else ""))

    def note(label: str) -> None:
        console.print(f"[yellow]•[/] {label}")

    console.print(f"[bold]shotvault {__version__}[/] · Python {sys.version.split()[0]} · {sys.platform}")
    console.print(f"config: {config_path(_state['config'])}")
    check(cfg.vault.is_dir(), f"vault {cfg.vault}", "folder not found")
    if (cfg.vault / ".git").exists():
        check(True, "vault is a git repository")
    elif cfg.git.commit:
        note("git.commit is on but the vault is not a git repository")
    free = shutil.disk_usage(cfg.data_dir if cfg.data_dir.exists() else Path.home()).free / 1e9
    check(free > 5, f"{free:.0f} GB free disk", "originals need ~1.5 GB per 1000 screenshots")

    if cfg.sources.photos.enabled:
        if sys.platform != "darwin":
            note("Photos source only works on macOS")
        else:
            try:
                import osxphotos  # noqa: F401

                check(True, "osxphotos installed")
            except ImportError:
                check(False, "osxphotos installed", "pip install 'shotvault[mac]'")
            library = Path(cfg.sources.photos.library or "~/Pictures/Photos Library.photoslibrary").expanduser()
            database = library / "database" / "Photos.sqlite"
            try:
                with database.open("rb"):
                    pass
                check(True, f"can read {library.name}")
            except PermissionError:
                check(
                    False,
                    "can read the Photos library",
                    "grant Full Disk Access to your terminal (System Settings → Privacy & Security)",
                )
            except FileNotFoundError:
                check(False, f"Photos library at {library}", "set sources.photos.library")
    if cfg.sources.iphone.enabled:
        try:
            import pymobiledevice3  # noqa: F401

            from shotvault.sources import iphone_connected

            connected = iphone_connected()
            check(True, "pymobiledevice3 installed")
            note("iPhone connected over USB" if connected else "no iPhone connected right now (that's fine)")
        except ImportError:
            check(False, "pymobiledevice3 installed", "pip install 'shotvault[iphone]'")
    for folder in cfg.sources.folders.paths if cfg.sources.folders.enabled else []:
        check(Path(folder).expanduser().is_dir(), f"folder {folder}", "not found")

    from shotvault.ocr import make_ocr

    engine = make_ocr(cfg.ocr)
    check(engine.name != "none", f"OCR engine: {engine.name}", "install shotvault[mac] (macOS) or pytesseract")

    llm = make_llm(cfg.llm)
    try:
        models = llm.models()
        check(True, f"{cfg.llm.provider} server at {cfg.llm.url}")
        wanted = {"vision": cfg.llm.vision_model, "text": cfg.llm.chat_model, "embedding": cfg.llm.embed_model}
        for role, model in wanted.items():
            if not model:
                continue
            present = any(m == model or m == f"{model}:latest" or m.split(":")[0] == model for m in models)
            hint = f"ollama pull {model}" if cfg.llm.provider == "ollama" else f"available: {', '.join(models[:8])}"
            check(present, f"{role} model {model}", hint)
    except Exception as exc:
        check(False, f"{cfg.llm.provider} server at {cfg.llm.url}", str(exc)[:120])

    if smoke and ok:
        _smoke(cfg)
    raise typer.Exit(0 if ok else 1)


def _smoke(cfg: Config) -> None:
    import tempfile

    from PIL import Image, ImageDraw

    from shotvault.db import OCR_DONE
    from shotvault.imaging import sha256_bytes

    image = Image.new("RGB", (1179, 2556), "white")
    draw = ImageDraw.Draw(image)
    lines = ["Notes", "Shopping list", "- oat milk", "- coffee beans (Ethiopia)", "Dentist Friday 10:30"]
    for i, text in enumerate(lines):
        draw.text((80, 200 + i * 120), text, fill="black", font_size=72)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "smoke.png"
        image.save(path)
        data = path.read_bytes()
        tmp_db = Database(Path(tmp) / "smoke.db")
        tmp_cfg = cfg.model_copy(update={"data_dir": Path(tmp)})
        pipe = Pipeline(tmp_cfg, tmp_db, console)
        sha = sha256_bytes(data)
        dest = tmp_cfg.originals_dir / sha[:2] / f"{sha}.png"
        dest.parent.mkdir(parents=True)
        dest.write_bytes(data)
        start = time.time()
        text = pipe.ocr.recognize(dest)
        shot = tmp_db.add_shot(
            id=sha[:12], sha256=sha, source="folder", original_name="smoke.png", ext=".png",
            width=1179, height=2556, status=OCR_DONE, ocr_text=text, taken_at=datetime.now().isoformat(),
        )  # fmt: skip
        with console.status("Running a test extraction…"):
            result = pipe.extract_one(shot)
        console.print(
            f"[green]✓[/] extraction in {time.time() - start:.1f}s: [bold]{result.title}[/] · {result.kind.value}"
        )
        console.print(f"  tags: {', '.join(result.tags)} · actions: {[a.task for a in result.actions]}")


# -- main loop -----------------------------------------------------------------
def _process(cfg: Config, db: Database, pipe: Pipeline, limit: Optional[int], render: bool) -> Counter:
    totals: Counter = Counter()
    totals.update(pipe.run_ocr(limit))
    try:
        totals.update(pipe.run_extract(limit))
    except LLMUnavailable as exc:
        console.print(f"[red]Stopped extraction: {exc}")
    if cfg.llm.embed_model:
        try:
            totals.update(pipe.run_embed())
        except LLMError as exc:
            console.print(f"[yellow]Embeddings skipped: {exc}")
    if render:
        stats = VaultRenderer(cfg, db, llm=pipe.llm, console=console).render()
        totals.update(
            {"notes_written": stats.written, "hubs_written": stats.hubs, "topics_synthesized": stats.topics_synthesized}
        )
        if stats.hand_edited:
            console.print(f"[dim]{stats.hand_edited} notes left alone (markers removed by you)")
    return totals


def _commit(cfg: Config, totals: Counter) -> None:
    if not cfg.git.commit:
        return
    from shotvault.macos import git_commit

    new = totals.get("extracted", 0)
    message = f"shotvault: {new} new screenshot{'s' if new != 1 else ''}" if new else "shotvault: update notes"
    if git_commit(cfg.vault, cfg.folder, message, push=cfg.git.push):
        console.print(f"[dim]committed to vault git: {message}")


@app.command()
def run(
    source: Optional[list[str]] = typer.Option(None, "--source", "-s", help="photos, iphone or folder (repeatable)."),
    limit: Optional[int] = typer.Option(None, help="Process at most N screenshots this run (newest first)."),
    no_ingest: bool = typer.Option(False, "--no-ingest", help="Only process what is already imported."),
    no_render: bool = typer.Option(False, "--no-render", help="Skip writing the vault."),
    quiet: bool = typer.Option(False, help="Plain output, for scheduled runs."),
) -> None:
    """Import new screenshots, understand them, and update the vault."""
    global console
    if quiet:
        console = Console(stderr=True, force_terminal=False, no_color=True)
    cfg = _cfg()
    with _locked(cfg):
        started = time.time()
        console.print(f"[dim]{datetime.now():%Y-%m-%d %H:%M} · vault {cfg.vault}")
        db = Database(cfg.db_path)
        pipe = Pipeline(cfg, db, console)
        totals: Counter = Counter()
        if not no_ingest:
            imported = pipe.ingest(source)
            console.print(f"Imported: {_fmt(imported)}")
            totals.update(imported)
        totals.update(_process(cfg, db, pipe, limit, render=not no_render))
        _commit(cfg, totals)
        minutes = (time.time() - started) / 60
        console.print(f"[bold green]Done[/] in {minutes:.1f} min · {_fmt(totals)}")


@app.command()
def ingest(source: Optional[list[str]] = typer.Option(None, "--source", "-s")) -> None:
    """Only import new screenshots (no model calls)."""
    cfg = _cfg()
    with _locked(cfg):
        console.print(f"Imported: {_fmt(Pipeline(cfg, Database(cfg.db_path), console).ingest(source))}")


@app.command()
def render(restore: bool = typer.Option(False, help="Recreate notes you deleted.")) -> None:
    """Regenerate the vault from the database (no model calls except topic synthesis)."""
    cfg = _cfg()
    with _locked(cfg):
        db = Database(cfg.db_path)
        stats = VaultRenderer(cfg, db, llm=make_llm(cfg.llm), console=console, restore=restore).render()
        console.print(
            f"{stats.written} notes written, {stats.unchanged} unchanged, {stats.hubs} hubs, "
            f"{stats.topics_synthesized} topics synthesized"
        )
        _commit(cfg, Counter())


@app.command()
def watch(
    interval: int = typer.Option(60, help="Seconds between checks."),
    photos_every: int = typer.Option(30, help="Minutes between Photos library scans."),
) -> None:
    """Stay running: import as soon as you plug in your iPhone or take a Mac screenshot."""
    from shotvault.sources import iphone_connected

    cfg = _cfg()
    console.print("[bold]Watching[/] (Ctrl-C to stop) · plug in your iPhone to import its screenshots")
    was_connected = False
    last_photos = 0.0
    try:
        while True:
            sources = ["folder"] if cfg.sources.folders.enabled else []
            connected = cfg.sources.iphone.enabled and iphone_connected()
            if connected and not was_connected:
                console.print("[cyan]iPhone connected")
                sources.append("iphone")
            was_connected = connected
            if cfg.sources.photos.enabled and time.time() - last_photos >= photos_every * 60:
                sources.append("photos")
                last_photos = time.time()
            if sources:
                with _locked(cfg, skip_if_busy=True) as acquired:
                    if not acquired:  # the nightly run is busy; try again next tick
                        time.sleep(interval)
                        continue
                    db = Database(cfg.db_path)
                    pipe = Pipeline(cfg, db, console)
                    imported = pipe.ingest(sources)
                    if imported.get("new"):
                        console.print(f"Imported: {_fmt(imported)}")
                        totals = _process(cfg, db, pipe, None, render=True)
                        _commit(cfg, totals)
                    db.close()
            time.sleep(interval)
    except KeyboardInterrupt:
        console.print("Stopped.")


# -- using the knowledge -----------------------------------------------------------
@app.command("search")
def search_cmd(
    query: list[str] = typer.Argument(..., help="What you remember about it."),
    limit: int = typer.Option(10, "--limit", "-n"),
    kind: Optional[str] = typer.Option(None, help="Filter: article, code, recipe, chat, place..."),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable output."),
    sensitive: bool = typer.Option(False, help="Include screenshots flagged sensitive."),
) -> None:
    """Hybrid keyword + semantic search."""
    from shotvault.search import obsidian_uri, search

    cfg = _cfg()
    db = Database(cfg.db_path)
    hits = search(
        db, cfg, " ".join(query), llm=make_llm(cfg.llm), limit=limit, kind=kind,
        include_sensitive=sensitive, warn=lambda m: console.print(f"[yellow]{m}"),
    )  # fmt: skip
    if as_json:
        out.print_json(json.dumps([h.as_dict(cfg) for h in hits], ensure_ascii=False))
        return
    if not hits:
        console.print("No matches.")
        return
    table = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
    for column in ("", "date", "kind", "title", "id"):
        table.add_column(column)
    for n, hit in enumerate(hits, 1):
        uri = obsidian_uri(cfg, hit.shot.note_path)
        title = rich_escape(hit.data.title)
        title = f"[link={uri}]{title}[/link]" if uri else title
        table.add_row(str(n), f"{hit.shot.taken:%Y-%m-%d}", KIND_LABELS[hit.data.kind], title, hit.shot.id[:8])
        if hit.data.summary:
            table.add_row("", "", "", f"[dim]{rich_escape(hit.data.summary[:140])}", "")
    out.print(table)


@app.command()
def ask(
    question: list[str] = typer.Argument(..., help="Ask anything about what you've screenshotted."),
    k: int = typer.Option(8, help="How many screenshots to read."),
    save: bool = typer.Option(False, help="Save the answer as a note in the vault."),
) -> None:
    """Answer a question from your screenshots, with sources."""
    from shotvault.search import answer, obsidian_uri

    cfg = _cfg()
    db = Database(cfg.db_path)
    text = " ".join(question)
    stream, hits = answer(db, cfg, make_llm(cfg.llm), text, k=k, warn=lambda m: console.print(f"[yellow]{m}"))
    chunks = []
    try:
        for piece in stream:
            chunks.append(piece)
            sys.stdout.write(piece)
            sys.stdout.flush()
    except LLMError as exc:
        console.print(f"\n[red]{exc}")
        raise typer.Exit(1) from exc
    sys.stdout.write("\n")
    if hits:
        console.print("\n[bold]Sources")
        for n, hit in enumerate(hits, 1):
            uri = obsidian_uri(cfg, hit.shot.note_path)
            label = rich_escape(hit.data.title)
            label = f"[link={uri}]{label}[/link]" if uri else label
            console.print(f" [{n}] {hit.shot.taken:%Y-%m-%d} · {label}", highlight=False)
    if save and hits:
        body = "".join(chunks).strip()
        names = {n: Path(h.shot.note_path).stem for n, h in enumerate(hits, 1) if h.shot.note_path}
        body = re.sub(r"\[(\d+)\]", lambda m: f"({link(names[int(m[1])], m[1])})" if int(m[1]) in names else m[0], body)
        path = cfg.root / "Answers" / f"{datetime.now():%Y-%m-%d} {safe_filename(text, 70)}.md"
        sources = [f"{n}. {link(names[n], hits[n - 1].data.title)}" for n in names]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {text}\n\n{body}\n\n## Sources\n" + "\n".join(sources) + "\n", encoding="utf-8")
        console.print(f"[green]Saved {path.relative_to(cfg.vault)}")


@app.command()
def show(
    shot_id: str = typer.Argument(..., help="Screenshot id (a prefix is enough)."),
    open_: bool = typer.Option(False, "--open", help="Open the original image."),
) -> None:
    """Everything shotvault knows about one screenshot."""
    cfg = _cfg()
    shot = Database(cfg.db_path).find(shot_id)
    if shot is None:
        console.print("[red]Not found.")
        raise typer.Exit(1)
    original = shot.original_path(cfg.originals_dir)
    out.print(
        f"[bold]{shot.id}[/] · {shot.status} · {shot.taken:%Y-%m-%d %H:%M} · {shot.source} · {shot.original_name}"
    )
    out.print(f"original: {original}")
    if shot.note_path:
        out.print(f"note: {shot.note_path}")
    if shot.duplicate_of:
        out.print(f"duplicate of: {shot.duplicate_of}")
    if shot.error:
        out.print(f"[red]last error: {shot.error}")
    if shot.data:
        out.print_json(shot.data.model_dump_json())
    if shot.ocr_text:
        out.print("[bold]OCR[/]")
        out.print(shot.ocr_text, markup=False, highlight=False)
    if open_ and sys.platform == "darwin":
        subprocess.run(["open", str(original)])


@app.command()
def reprocess(
    ids: Optional[list[str]] = typer.Argument(None, help="Screenshot ids."),
    all_: bool = typer.Option(False, "--all", help="Everything."),
    kind: Optional[str] = typer.Option(None, help="Only this kind."),
    outdated: bool = typer.Option(False, help="Only those made with another model or prompt version."),
    errors: bool = typer.Option(False, help="Retry failed ones."),
    ocr: bool = typer.Option(False, help="Also redo OCR and duplicate detection."),
) -> None:
    """Queue screenshots for extraction again (e.g. after switching to a better model)."""
    cfg = _cfg()
    db = Database(cfg.db_path)
    pipe = Pipeline(cfg, db, console)
    if ids:
        shots = [s for s in (db.find(i) for i in ids) if s]
    elif errors:
        shots = db.shots(ERROR)
    elif all_ or kind or outdated:
        shots = db.shots(EXTRACTED, ERROR, DUPLICATE) if ocr else db.shots(EXTRACTED, ERROR)
        if kind:
            shots = [s for s in shots if s.data and s.data.kind.value == kind]
        if outdated:
            shots = [s for s in shots if s.extractor != pipe.extractor_id]
    else:
        console.print("Say what to reprocess: ids, --all, --kind, --outdated or --errors.")
        raise typer.Exit(1)
    count = pipe.reset(shots, redo_ocr=ocr)
    console.print(f"{count} screenshots queued. Run `shotvault run --no-ingest` to process them.")


@app.command()
def stats() -> None:
    """Numbers about your screenshot knowledge base."""
    cfg = _cfg()
    db = Database(cfg.db_path)
    counts = db.counts()
    shots = db.shots(EXTRACTED)
    data = [s.data for s in shots if s.data]
    table = Table(title="shotvault", show_header=False, box=None)
    table.add_row("screenshots", str(sum(counts.values())))
    for status, n in sorted(counts.items()):
        table.add_row(f"  {status}", str(n))
    table.add_row("sources", ", ".join(f"{k} {v}" for k, v in Counter(s.source for s in db.iter_all()).items()))
    table.add_row("sensitive", str(sum(d.sensitive for d in data)))
    table.add_row("temporary", str(sum(d.keep_value == KeepValue.EPHEMERAL for d in data)))
    table.add_row("kinds", ", ".join(f"{k.value} {n}" for k, n in Counter(d.kind for d in data).most_common(8)))
    table.add_row(
        "top tags", ", ".join(f"#{t} {n}" for t, n in Counter(t for d in data for t in d.tags).most_common(12))
    )
    size = (
        sum(p.stat().st_size for p in cfg.originals_dir.rglob("*") if p.is_file()) if cfg.originals_dir.exists() else 0
    )
    table.add_row("originals", f"{size / 1e9:.2f} GB in {cfg.originals_dir}")
    out.print(table)


@app.command("cleanup-album")
def cleanup_album(
    name: str = typer.Option("shotvault · safe to delete", help="Album name in Photos."),
    duplicates: bool = typer.Option(True, help="Include duplicates."),
    dry_run: bool = typer.Option(False, help="Only count."),
) -> None:
    """Collect temporary and duplicate screenshots in a Photos album; review and delete them on your iPhone."""
    if sys.platform != "darwin":
        console.print("[red]Needs macOS Photos.")
        raise typer.Exit(1)
    cfg = _cfg()
    db = Database(cfg.db_path)
    extracted = db.shots(EXTRACTED)
    temporary = {s.id for s in extracted if s.data and s.data.keep_value == KeepValue.EPHEMERAL}
    dups = {s.id for s in db.shots(DUPLICATE)} if duplicates else set()
    refs = db.refs_for("photos", [s.id for s in extracted] + list(dups))
    uuids: list[str] = []
    for shot_id, shot_refs in refs.items():
        if shot_id in temporary or shot_id in dups:
            uuids += shot_refs
        elif duplicates:
            uuids += shot_refs[1:]  # byte-identical copies in the library
    if dry_run or not uuids:
        console.print(f"{len(uuids)} Photos items would be added to “{name}”.")
        return
    from shotvault.macos import add_to_album

    added = add_to_album(uuids, name)
    console.print(f"[green]Added {added} screenshots to the Photos album “{name}”. Nothing was deleted.")


@app.command()
def mcp() -> None:
    """Serve your screenshots to MCP clients (LM Studio, Claude Desktop…) over stdio."""
    from shotvault.mcp_server import serve

    serve(_cfg())


@schedule_app.command("install")
def schedule_install(
    hour: int = typer.Option(3, help="Hour of day (local time)."),
    minute: int = typer.Option(17, help="Minute."),
) -> None:
    """Run every night (keeps the Mac awake while it works)."""
    from shotvault.macos import install_schedule

    cfg = _cfg()
    plist = install_schedule(cfg, config_path(_state["config"]), hour, minute)
    console.print(f"[green]Scheduled daily at {hour:02d}:{minute:02d}[/] ({plist})")
    console.print(f"Logs: {cfg.logs_dir / 'nightly.log'}")
    console.print(
        "[dim]If the Photos source fails in the nightly log, give Full Disk Access to the Python "
        f"that runs shotvault: {Path(sys.executable).resolve()}"
    )


@schedule_app.command("remove")
def schedule_remove() -> None:
    """Remove the nightly job."""
    from shotvault.macos import uninstall_schedule

    console.print("Removed." if uninstall_schedule() else "No schedule installed.")


if __name__ == "__main__":  # pragma: no cover
    app()
