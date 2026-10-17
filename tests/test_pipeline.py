from __future__ import annotations

import yaml
from PIL import Image
from rich.console import Console

from conftest import FakeLLM, FakeOCR, make_screenshot
from shotvault.db import DUPLICATE, EXTRACTED
from shotvault.pipeline import Pipeline
from shotvault.search import search
from shotvault.vault import END, VaultRenderer, split_frontmatter

RUST = "@ferris · 2h\nTIL tokio::select! drops the losing branch\nRust async is subtle\ndocs.rs/tokio"
RECIPE = "Banana bread\n3 ripe bananas\n250 g flour\nBake 50 min at 180°C"
CODE = "Your verification code is 482913. Do not share it."


def setup_inbox(cfg, ocr: FakeOCR):
    inbox = cfg.sources.folders.paths[0]
    from pathlib import Path

    inbox = Path(inbox)
    rust = make_screenshot(inbox / "IMG_0001.PNG", RUST.splitlines())
    recipe = make_screenshot(inbox / "IMG_0002.PNG", RECIPE.splitlines(), color="#ffe")
    code = make_screenshot(inbox / "IMG_0003.PNG", CODE.split(". "), color="#eef")
    make_screenshot(inbox / "holiday.png", ["beach"], marker=False)  # not a screenshot
    # the Rust screenshot again, re-encoded (e.g. sent through a chat app)
    copy = inbox / "Screenshot 2025-03-14 at 21.07.12.jpg"
    Image.open(rust).convert("RGB").save(copy, quality=75)
    for path, text in [(rust, RUST), (recipe, RECIPE), (code, CODE), (copy, RUST)]:
        ocr.register(path, text)
    return inbox


def run_all(cfg, db, llm, ocr):
    pipe = Pipeline(cfg, db, Console(quiet=True), llm=llm, ocr=ocr)
    imported = pipe.ingest()
    ocr_stats = pipe.run_ocr()
    extracted = pipe.run_extract()
    embedded = pipe.run_embed()
    stats = VaultRenderer(cfg, db, llm=llm, console=Console(quiet=True)).render()
    return pipe, imported, ocr_stats, extracted, embedded, stats


def note_for(db, cfg, title_part):
    for shot in db.shots(EXTRACTED):
        if title_part in (shot.data.title if shot.data else ""):
            return cfg.vault / shot.note_path, shot
    raise AssertionError(title_part)


def test_end_to_end(cfg, db):
    ocr, llm = FakeOCR(), FakeLLM(cfg)
    setup_inbox(cfg, ocr)
    pipe, imported, ocr_stats, extracted, embedded, stats = run_all(cfg, db, llm, ocr)

    assert imported["new"] == 4 and imported["not_screenshots"] == 1
    assert ocr_stats["duplicates"] == 1
    assert extracted["extracted"] == 3 and llm.vision_calls == 3
    assert embedded["embedded"] == 3
    dup = db.shots(DUPLICATE)[0]
    assert dup.original_name.endswith(".jpg")
    assert dup.taken_at == "2025-03-14T21:07:12"  # from the file name

    # the prompt carries metadata and OCR
    assert any("device: phone" in p and "tokio::select!" in p for p in llm.prompts)

    note, shot = note_for(db, cfg, "Tokio")
    text = note.read_text()
    fm, body = split_frontmatter(text)
    assert fm["shotvault"] == shot.id
    assert fm["kind"] == "social_post"
    assert fm["tags"] == ["screenshot", "screenshot/social-post", "rust", "async-programming"]
    assert "[[Tokio]]" in fm["entities"]
    assert fm["urls"] == ["docs.rs/tokio"]  # hallucinated URL dropped (not in OCR)
    assert f"![[{shot.id}.webp|420]]" in body
    assert (cfg.root / "_attachments" / f"{shot.taken:%Y}" / f"{shot.taken:%m}" / f"{shot.id}.webp").exists()
    assert "```rust\ntokio::select!" in body
    assert "> [!quote]- Text in the screenshot" in body
    assert body.rstrip().endswith("## Notes")
    # entities merged case-insensitively across screenshots ("rust" / "Rust")
    assert "[[Rust]]" in body or "[[rust]]" in body
    assert (cfg.root / "Entities").exists()

    # sensitive screenshot is redacted: no image, no code in the vault
    private, private_shot = next((cfg.vault / s.note_path, s) for s in db.shots(EXTRACTED) if s.data.sensitive)
    private_text = private.read_text()
    assert "482913" not in private_text and "482913" not in private.name
    assert "Sensitive screenshot" in private_text
    assert not list(cfg.root.rglob(f"{private_shot.id}.webp"))

    # hubs
    home = (cfg.root / "Screenshots Home.md").read_text()
    assert "3 screenshots understood" in home
    actions = (cfg.root / "Screenshot Actions.md").read_text()
    assert "- [ ] Audit select! usages 📅 2025-03-20" in actions
    cleanup = (cfg.root / "Screenshot Cleanup.md").read_text()
    assert "21.07.12.jpg" in cleanup and "IMG_0003.PNG" in cleanup
    topic = (cfg.root / "Topics" / "Rust (screenshots).md").read_text()
    assert "## Overview" in topic and "[[" in topic
    assert llm.chat_calls == 1
    assert list((cfg.root / "Months").glob("Screenshots *.md"))

    # search: keywords and semantic
    hits = search(db, cfg, "tokio cancellation", llm=llm)
    assert hits and hits[0].shot.id == shot.id
    assert all(not h.data.sensitive for h in search(db, cfg, "verification code bank", llm=llm))


def test_rerender_is_idempotent_and_keeps_your_edits(cfg, db):
    ocr, llm = FakeOCR(), FakeLLM(cfg)
    setup_inbox(cfg, ocr)
    run_all(cfg, db, llm, ocr)
    note, shot = note_for(db, cfg, "Tokio")

    # nothing changes on a second render (and topic synthesis is cached)
    stats = VaultRenderer(cfg, db, llm=llm, console=Console(quiet=True)).render()
    assert stats.written == 0 and stats.hubs == 0
    assert llm.chat_calls == 1

    # your notes below the marker, your own properties and tags survive
    fm, body = split_frontmatter(note.read_text())
    fm["rating"] = 5
    fm["tags"].append("to-review")
    body += "\nMy own thoughts about this.\n"
    moved = cfg.vault / "Inbox" / "renamed by me.md"
    moved.parent.mkdir()
    moved.write_text("---\n" + yaml.safe_dump(fm, sort_keys=False, allow_unicode=True) + "---\n" + body)
    note.unlink()

    # ticking an action is kept
    actions_path = cfg.root / "Screenshot Actions.md"
    actions_path.write_text(actions_path.read_text().replace("- [ ] Audit", "- [x] Audit"))

    VaultRenderer(cfg, db, llm=llm, console=Console(quiet=True)).render()
    assert not note.exists(), "moved note must be found by id, not recreated"
    fm2, body2 = split_frontmatter(moved.read_text())
    assert fm2["rating"] == 5
    assert "to-review" in fm2["tags"] and "rust" in fm2["tags"]
    assert "My own thoughts about this." in body2.split(END, 1)[1]
    assert "- [x] Audit select! usages" in actions_path.read_text()

    # removing the markers hands the note over to you
    moved.write_text("---\nshotvault: " + shot.id + "\n---\nAll mine now.\n")
    stats = VaultRenderer(cfg, db, llm=llm, console=Console(quiet=True)).render()
    assert stats.hand_edited == 1
    assert moved.read_text().endswith("All mine now.\n")


def test_incremental_ingest_skips_known_files(cfg, db):
    ocr, llm = FakeOCR(), FakeLLM(cfg)
    setup_inbox(cfg, ocr)
    run_all(cfg, db, llm, ocr)
    pipe = Pipeline(cfg, db, Console(quiet=True), llm=llm, ocr=ocr)
    again = pipe.ingest()
    assert again.get("new", 0) == 0 and again.get("not_screenshots", 0) == 0
    assert pipe.run_extract().get("extracted", 0) == 0


def test_skip_mode_writes_no_note_for_sensitive(cfg, db):
    cfg.privacy.sensitive = "skip"
    ocr, llm = FakeOCR(), FakeLLM(cfg)
    setup_inbox(cfg, ocr)
    run_all(cfg, db, llm, ocr)
    notes = [p for p in cfg.root.rglob("*.md") if "Sensitive" in p.read_text() or "482913" in p.read_text()]
    assert notes == []


def test_deleted_notes_stay_deleted_until_restore(cfg, db):
    ocr, llm = FakeOCR(), FakeLLM(cfg)
    setup_inbox(cfg, ocr)
    run_all(cfg, db, llm, ocr)
    note, shot = note_for(db, cfg, "Banana")
    note.unlink()
    stats = VaultRenderer(cfg, db, llm=llm, console=Console(quiet=True)).render()
    assert stats.deleted_by_you == 1 and not note.exists()
    assert "Banana" not in (cfg.root / "Screenshots Home.md").read_text()
    VaultRenderer(cfg, db, llm=llm, console=Console(quiet=True), restore=True).render()
    assert note.exists()


def test_note_flagged_sensitive_later_is_renamed_and_redacted(cfg, db):
    ocr, llm = FakeOCR(), FakeLLM(cfg)
    setup_inbox(cfg, ocr)
    cfg.privacy.sensitive = "keep"
    run_all(cfg, db, llm, ocr)
    note, shot = note_for(db, cfg, "Bank code")
    assert "482913" in note.name
    cfg.privacy.sensitive = "redact"
    VaultRenderer(cfg, db, llm=llm, console=Console(quiet=True)).render()
    assert not note.exists()
    renamed = cfg.vault / db.get(shot.id).note_path
    assert "Private" in renamed.name and "482913" not in renamed.read_text()
    assert not list(cfg.root.rglob(f"{shot.id}.webp"))
