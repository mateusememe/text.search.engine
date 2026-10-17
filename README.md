# shotvault

**Your screenshots, turned into a searchable Obsidian knowledge base by models running on your own Mac.**

Every screenshot you take is a small "remember this": a recipe, a tweet thread, an error message, a place, a product, an idea. shotvault picks them all up from **iCloud Photos**, an **iPhone plugged in over USB**, or your **Desktop**. It reads them with **Apple Vision OCR**, has a local vision model (via **Ollama** or **LM Studio**) work out what each one means, and writes them into your vault as linked notes. You can then search them, ask questions about them, or chat with them from LM Studio over MCP.

Nothing leaves your machine.

```
 iCloud Photos ─┐                                                              ┌─ one note per screenshot
 iPhone (USB) ──┼─► import ─► OCR ─► vision model ─► embeddings ─► Obsidian ───┼─ topics · entities · actions
 ~/Desktop ─────┘   dedupe   Apple    (Ollama /     (search)       vault (git) └─ cleanup · on this day · months
                             Vision   LM Studio)
                                         │
                       shotvault search · shotvault ask · MCP server (LM Studio / Claude Desktop)
```

## What lands in your vault

```
Screenshots/
├── Screenshots Home.md            dashboard: recent, topics, counts by kind, months
├── Screenshot Actions.md          every to-do found in screenshots, as checkboxes (your ticks are kept)
├── Screenshot Cleanup.md          temporary screenshots and duplicates that are safe to delete
├── Screenshots On This Day.md     what you saved on this date in earlier years, plus a few to rediscover
├── 2025/03/2025-03-14 Tokio select! cancellation gotcha.md
├── Topics/Rust (screenshots).md   a model-written summary of everything you saved about #rust, with citations
├── Entities/Tokio.md              people, products and tools mentioned in 2+ screenshots
├── Months/Screenshots 2025-03.md
└── _attachments/2025/03/deca9c3481e3.webp
```

A note looks like this (shortened; the frontmatter feeds Obsidian properties and Dataview):

```markdown
---
shotvault: deca9c3481e3
title: Tokio select! cancellation gotcha
kind: social_post
app: X
captured: 2025-03-14T21:07
tags: [screenshot, screenshot/social-post, rust, async-programming]
entities: ["[[Tokio]]", "[[Rust]]"]
keep: reference
---
# Tokio select! cancellation gotcha
![[deca9c3481e3.webp|420]]
> [!summary]
> Losing branches in tokio::select! are dropped mid-await, so futures must be cancellation safe.

**What it shows:** A tweet thread about Rust async cancellation.
**Why you saved it:** Reference for a cancellation bug.
### Key points · ### Code · ### To do · ### Links
### Connections   Mentions: [[Tokio]] · [[Rust]]   Topics: [[Rust (screenshots)|Rust]]   Same moment: [[…]]
> [!quote]- Text in the screenshot   (collapsed full OCR text)

## Notes
(anything you write here is yours forever)
```

Entities are plain `[[wikilinks]]`. If your vault already has a `Rust.md`, screenshots link straight into **your** existing note, and shotvault doesn't create its own.

## Quick start (MacBook Pro M3 Pro, 18 GB)

```bash
# 1. a local model server: pick one
brew install ollama && ollama pull qwen2.5vl:7b && ollama pull bge-m3
#   …or LM Studio: download "Qwen2.5-VL-7B-Instruct" (MLX 4-bit is fastest on Apple Silicon)
#   plus an embedding model, then run `lms server start`

# 2. install shotvault
git clone https://github.com/mateusememe/text.search.engine && cd text.search.engine
uv tool install --editable ".[all]"        # or: python3 -m venv .venv && .venv/bin/pip install -e ".[all]"

# 3. point it at your vault, then check everything
shotvault init ~/Documents/MyVault           # add --provider openai for LM Studio
shotvault doctor --smoke                     # checks permissions and models, runs one test extraction

# 4. try it on your 20 newest screenshots, then open Obsidian
shotvault run --limit 20

# 5. process the whole backlog (resumable; Ctrl-C or closing the lid is fine)
shotvault run
shotvault schedule install                   # then every night at 03:17
```

**macOS permissions:**
- Give your terminal app **Full Disk Access** so it can read the Photos library (System Settings → Privacy & Security). `doctor` tells you if it's missing.
- The first iCloud download asks for permission to control Photos.
- For the nightly job, give Full Disk Access to the Python path that `shotvault schedule install` prints.

## Sources

| Source | What it covers | Notes |
|---|---|---|
| `photos` | Everything in **iCloud Photos** (Apple's Screenshots album) | Uses [osxphotos](https://github.com/RhetTbull/osxphotos). Originals stored only in iCloud are downloaded through Photos (`download_missing`). Keeps albums, place and your captions as context. |
| `iphone` | An **iPhone over USB** (unlock it and tap *Trust*) | Uses [pymobiledevice3](https://github.com/doronz88/pymobiledevice3). Only PNGs with the iOS screenshot marker are imported; camera photos are never copied. With *Optimize iPhone Storage* on, only originals still on the phone are visible; the Photos source covers the rest. |
| `folder` | `~/Desktop` (macOS screenshots), AirDrop folders, anything else | Matches the screenshot marker or localised names (*Screenshot…*, *Captura de Tela…*). |

All sources are **read-only**: nothing is ever deleted from Photos, the phone or your folders. The same screenshot arriving from two sources is stored once (content hash). Re-captures of the same screen are caught as near-duplicates (perceptual hash, confirmed by matching OCR text).

```bash
shotvault run -s iphone        # only the phone that's plugged in
shotvault watch                # stay running: imports when you plug the iPhone in or take a Mac screenshot
```

## Using your knowledge

```bash
shotvault search banana bread recipe        # hybrid keyword + semantic; titles are clickable obsidian:// links
shotvault search "rust async" --kind code --json
shotvault ask "what was that terminal command to resize images?"   # answers with [1] sources
shotvault ask "which restaurants did I save in Lisbon?" --save      # saves the answer as a note
shotvault show deca9c --open                 # everything about one screenshot, and opens the original
shotvault stats
```

**Chat with your screenshots in LM Studio or Claude Desktop.** Add this to LM Studio's `mcp.json` (or `claude_desktop_config.json`):

```json
{ "mcpServers": { "shotvault": { "command": "shotvault", "args": ["mcp"] } } }
```

GUI apps may not see your shell's `PATH`; if the server doesn't start, use the full path from `which shotvault`. Tools exposed: `search_screenshots`, `get_screenshot` (full OCR text), `view_screenshot` (the image itself), `recent_screenshots`.

**Clean up your camera roll.** `shotvault cleanup-album` puts every screenshot the model judged temporary (one-time codes, delivery updates, used tickets) and every duplicate into a Photos album called *shotvault · safe to delete*. The album syncs to your iPhone, where you can review it and delete in one go. shotvault itself never deletes anything.

## Models on 18 GB of unified memory

| Role | Ollama | LM Studio | Memory |
|---|---|---|---|
| Vision, **default** | `qwen2.5vl:7b` | Qwen2.5-VL-7B-Instruct (MLX 4-bit) | ~6 GB |
| Vision, faster | `qwen2.5vl:3b` | Qwen2.5-VL-3B-Instruct | ~3 GB |
| Vision, worth trying | `qwen3-vl:8b`, `gemma3:12b` | same models | 6–9 GB |
| Embeddings | `bge-m3` (multilingual, good for PT + EN) | `text-embedding-nomic-embed-text-v1.5` | ~1.2 GB |
| Topic summaries and `ask` (optional) | `qwen3:8b`, or the vision model | any instruct model | ~5 GB |

Rough expectation (an estimate, not a benchmark): a 7B vision model takes about **20–40 s per screenshot** on an M3 Pro, so ~3,000 screenshots is one or two nights. `run --limit 20` shows elapsed time and an ETA, so you'll know your real speed in a minute. Every stage saves after each screenshot, and runs resume where they stopped.

Switched to a better model later? `shotvault reprocess --outdated && shotvault run --no-ingest` re-reads only the screenshots made with the old model or prompt. Your notes stay where they are.

## Privacy

- Everything runs locally: Apple's on-device Vision OCR, and models through Ollama or LM Studio on localhost.
- The model flags screenshots showing passwords, one-time codes, bank details, IDs, medical or intimate content. With `privacy.sensitive = "redact"` (default), those notes contain no image and no text, only a pointer to `shotvault show <id>`, because your vault syncs through Git. Use `"skip"` for no note at all, or `"keep"` to treat them like any other screenshot.
- Originals and the database live in `~/.shotvault`, outside the vault.

## A vault synced with Git

- With `git.commit = true`, each run makes one commit that touches only `Screenshots/`. Pushing is left to Obsidian Git unless `git.push = true`.
- Images add up: about 250 KB per screenshot as WebP, roughly 750 MB per 3,000. If that's too much for your repo (or for Obsidian Git on the iPhone), do one of these:
  - add `Screenshots/_attachments/` to the vault's `.gitignore` (images stay on the Mac),
  - set `attachments.quality` / `max_side` lower, or
  - set `attachments.mode = "none"` for a text-only vault.

## Your notes stay yours

- shotvault only rewrites the block between its `shotvault:start` / `shotvault:end` markers and the frontmatter keys it owns.
- Text below the end marker, extra properties and tags you add survive every run.
- Rename or move notes freely: they're found by their `shotvault:` id.
- Delete the markers from a note and shotvault never touches it again.
- Delete a note and it stays deleted (`shotvault render --restore` brings deleted notes back).
- Ticks in *Screenshot Actions* are kept.
- Files are only rewritten when their content changes, so Git history stays quiet.

## Configuration

`shotvault init` writes `~/.config/shotvault/config.toml`. Every option is listed there with a comment; the full schema is in [`src/shotvault/config.py`](src/shotvault/config.py). Set `language = "Português (Brasil)"` to get notes written in Portuguese.

## How it's built

The design is ports and adapters: each source, OCR engine and model provider sits behind a small interface, so swapping Ollama for LM Studio, or Vision for Tesseract, is a config change.

| Module | Role |
|---|---|
| `sources.py` | Photos (osxphotos), iPhone (pymobiledevice3, streamed from a worker thread), folders |
| `ocr.py` | Apple Vision / Live Text (ocrmac), Tesseract fallback, reading-order reconstruction |
| `llm.py` | Ollama native API and OpenAI-compatible API, schema-constrained JSON, streaming, embeddings |
| `models.py` | The `Extraction` schema: lenient parsing of model output plus a strict wire schema for grammar-constrained decoding |
| `pipeline.py` | ingest → OCR + near-duplicates → extraction (grounded against OCR) → embeddings |
| `db.py` | SQLite: state, FTS5 full-text index, embedding vectors, topic summary cache |
| `vault.py` | Markdown rendering with the ownership rules above; hubs |
| `search.py` | FTS5 BM25 + cosine similarity fused with reciprocal rank fusion; `ask` |
| `mcp_server.py`, `macos.py` | MCP tools; launchd schedule, Photos album, git |

The vault is a rendering of the database, so changing templates or models never needs a re-import.

## Development

```bash
pip install -e ".[dev,iphone,mcp]"
pytest          # fake model and fake OCR run the whole pipeline end to end; Vision OCR tests run on macOS
ruff check src tests
```

## Ideas for later

- Group a series of screenshots of one long thread into a single note.
- Weekly digest notes written by the text model.
- Spaced-repetition review of `reference` screenshots.
- Screen recordings: sample frames and summarise them.
