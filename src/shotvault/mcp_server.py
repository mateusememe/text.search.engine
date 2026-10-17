"""MCP server so LM Studio, Claude Desktop or any MCP client can search your
screenshots while you chat. Run with `shotvault mcp` (stdio)."""

from __future__ import annotations

from datetime import datetime, timedelta

from shotvault.config import Config
from shotvault.db import EXTRACTED, Database
from shotvault.imaging import fit, open_image
from shotvault.llm import make_llm
from shotvault.search import obsidian_uri, search


def build_server(cfg: Config):
    try:
        from mcp.server.mcpserver import Image, MCPServer as Server  # mcp >= 2
    except ImportError:
        from mcp.server.fastmcp import FastMCP as Server, Image  # mcp 1.x

    db = Database(cfg.db_path)
    llm = make_llm(cfg.llm)
    server = Server(
        "shotvault",
        instructions=(
            "Search the user's screenshot knowledge base (OCR text, summaries, tags) "
            "and cite results with their obsidian_url."
        ),
    )

    @server.tool()
    def search_screenshots(query: str, limit: int = 8, kind: str | None = None) -> list[dict]:
        """Hybrid keyword + semantic search over the user's screenshots.
        `kind` filters by category (article, code, chat, recipe, shopping, place, event...)."""
        return [hit.as_dict(cfg) for hit in search(db, cfg, query, llm=llm, limit=limit, kind=kind)]

    @server.tool()
    def get_screenshot(shot_id: str) -> dict:
        """Full details of one screenshot, including its OCR text."""
        shot = db.find(shot_id)
        if shot is None or shot.data is None:
            return {"error": f"no screenshot {shot_id}"}
        data = shot.data
        if data.sensitive and cfg.privacy.sensitive != "keep":
            return {"id": shot.id, "sensitive": True, "note": "Sensitive screenshot; not shared."}
        return {
            **data.model_dump(mode="json"),
            "id": shot.id,
            "captured": shot.taken.isoformat(timespec="minutes"),
            "ocr_text": shot.ocr_text,
            "note": shot.note_path,
            "obsidian_url": obsidian_uri(cfg, shot.note_path),
        }

    @server.tool()
    def view_screenshot(shot_id: str):
        """The screenshot image itself (downscaled), for visual questions."""
        shot = db.find(shot_id)
        if shot is None or (shot.data and shot.data.sensitive and cfg.privacy.sensitive != "keep"):
            return "Not available."
        import io

        image = fit(open_image(shot.original_path(cfg.originals_dir).read_bytes()), 1_500_000, 2048)
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, "JPEG", quality=85)
        return Image(data=buffer.getvalue(), format="jpeg")

    @server.tool()
    def recent_screenshots(days: int = 7, limit: int = 30) -> list[dict]:
        """Screenshots captured in the last `days` days, newest first."""
        since = datetime.now() - timedelta(days=days)
        out = []
        for shot in db.shots(EXTRACTED):
            if shot.taken < since:
                break
            data = shot.data
            if data is None or (data.sensitive and cfg.privacy.sensitive != "keep"):
                continue
            out.append(
                {
                    "id": shot.id,
                    "title": data.title,
                    "kind": data.kind.value,
                    "captured": shot.taken.isoformat(timespec="minutes"),
                    "summary": data.summary,
                    "obsidian_url": obsidian_uri(cfg, shot.note_path),
                }
            )
            if len(out) >= limit:
                break
        return out

    return server


def serve(cfg: Config) -> None:
    build_server(cfg).run()
