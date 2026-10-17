import asyncio
import json

from typer.testing import CliRunner

from conftest import FakeLLM, FakeOCR
from shotvault import cli
from shotvault.config import load_config
from shotvault.db import Database
from test_pipeline import run_all, setup_inbox

runner = CliRunner()


def test_init_writes_a_valid_config(tmp_path):
    vault = tmp_path / "Vault"
    (vault / ".obsidian").mkdir(parents=True)
    target = tmp_path / "config.toml"
    result = runner.invoke(cli.app, ["--config", str(target), "init", str(vault), "--provider", "openai"])
    assert result.exit_code == 0, result.output
    cfg = load_config(target)
    assert cfg.vault == vault.resolve()
    assert cfg.llm.provider == "openai" and cfg.llm.url == "http://localhost:1234/v1"
    assert cfg.llm.embed_model == "text-embedding-nomic-embed-text-v1.5"
    again = runner.invoke(cli.app, ["--config", str(target), "init", str(vault)])
    assert again.exit_code == 1  # never overwrites without --force


def _populated(tmp_path, cfg, monkeypatch):
    ocr, llm = FakeOCR(), FakeLLM(cfg)
    setup_inbox(cfg, ocr)
    db = Database(cfg.db_path)
    run_all(cfg, db, llm, ocr)
    db.close()
    monkeypatch.setattr(cli, "make_llm", lambda _: llm)
    monkeypatch.setattr(cli, "_cfg", lambda: cfg)
    return llm


def test_search_json_and_stats(tmp_path, cfg, monkeypatch):
    _populated(tmp_path, cfg, monkeypatch)
    result = runner.invoke(cli.app, ["search", "banana", "bread", "--json"])
    assert result.exit_code == 0, result.output
    hits = json.loads(result.stdout)
    assert hits[0]["title"].startswith("Banana bread")
    assert hits[0]["obsidian_url"].startswith("obsidian://open?vault=vault&file=Screenshots")
    stats = runner.invoke(cli.app, ["stats"])
    assert stats.exit_code == 0 and "sensitive" in stats.stdout


def test_ask_saves_an_answer_note(tmp_path, cfg, monkeypatch):
    _populated(tmp_path, cfg, monkeypatch)
    result = runner.invoke(cli.app, ["ask", "what", "about", "tokio", "--save"])
    assert result.exit_code == 0, result.output
    saved = list((cfg.root / "Answers").glob("*.md"))
    assert saved and "[[" in saved[0].read_text()


def test_reprocess_queues_outdated(tmp_path, cfg, monkeypatch):
    _populated(tmp_path, cfg, monkeypatch)
    cfg.llm.vision_model = "qwen3-vl:8b"  # "upgrade" the model
    result = runner.invoke(cli.app, ["reprocess", "--outdated"])
    assert result.exit_code == 0
    assert "3 screenshots queued" in result.stderr


def test_mcp_server_exposes_tools(cfg):
    from shotvault.mcp_server import build_server

    cfg.llm.embed_model = None  # keyword search only, no server needed
    server = build_server(cfg)
    names = {tool.name for tool in asyncio.run(server.list_tools())}
    assert {"search_screenshots", "get_screenshot", "view_screenshot", "recent_screenshots"} <= names
