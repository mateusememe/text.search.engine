"""Local model clients: Ollama's native API, and the OpenAI-compatible API that
LM Studio (and llama.cpp server, MLX, vLLM...) expose."""

from __future__ import annotations

import base64
import json
import re
import time
from typing import Any, Iterator, Protocol

import httpx

from shotvault.config import LLMConfig


class LLMError(RuntimeError):
    pass


class LLMUnavailable(LLMError):
    """Server down or model missing: stop the run instead of failing every screenshot."""


class LLM(Protocol):
    cfg: LLMConfig

    def vision_json(self, system: str, prompt: str, image: bytes, schema: dict[str, Any]) -> str: ...

    def chat(self, system: str, prompt: str, stream: bool = False) -> Iterator[str]: ...

    def embed(self, texts: list[str]) -> list[list[float]]: ...

    def models(self) -> list[str]: ...


_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_thinking(text: str) -> str:
    return _THINK.sub("", text).strip()


def _with_retries(call, attempts: int = 3):
    delay = 2.0
    for attempt in range(attempts):
        try:
            return call()
        except httpx.TimeoutException as exc:
            if attempt == attempts - 1:
                raise LLMError(f"timed out: {exc}") from exc
        except httpx.TransportError as exc:
            if attempt == attempts - 1:
                raise LLMUnavailable(f"cannot reach the model server ({exc}); is Ollama / LM Studio running?") from exc
        except httpx.HTTPStatusError as exc:
            status, detail = exc.response.status_code, exc.response.text[:500]
            if status == 404:
                raise LLMUnavailable(f"model or endpoint not found: {detail}") from exc
            if status < 500 or attempt == attempts - 1:
                raise LLMError(f"HTTP {status}: {detail}") from exc
        time.sleep(delay)
        delay *= 2


class OllamaClient:
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self.http = httpx.Client(base_url=cfg.url, timeout=cfg.timeout)

    def _options(self) -> dict[str, Any]:
        return {"temperature": self.cfg.temperature, "num_ctx": self.cfg.num_ctx, "num_predict": self.cfg.max_tokens}

    def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        def call():
            response = self.http.post(path, json=body)
            response.raise_for_status()
            return response.json()

        return _with_retries(call)

    def vision_json(self, system: str, prompt: str, image: bytes, schema: dict[str, Any]) -> str:
        body = {
            "model": self.cfg.vision_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt, "images": [base64.b64encode(image).decode()]},
            ],
            "format": schema,
            "stream": False,
            "keep_alive": self.cfg.keep_alive,
            "options": self._options(),
        }
        if self.cfg.think is not None:
            body["think"] = self.cfg.think
        return strip_thinking(self._post("/api/chat", body)["message"]["content"])

    def chat(self, system: str, prompt: str, stream: bool = False) -> Iterator[str]:
        body: dict[str, Any] = {
            "model": self.cfg.chat_model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            "stream": stream,
            "keep_alive": self.cfg.keep_alive,
            "options": {**self._options(), "temperature": max(self.cfg.temperature, 0.3)},
        }
        if self.cfg.think is not None:
            body["think"] = self.cfg.think
        if not stream:
            yield strip_thinking(self._post("/api/chat", body)["message"]["content"])
            return
        in_think = False
        with self.http.stream("POST", "/api/chat", json=body) as response:
            if response.status_code >= 400:
                response.read()
                raise LLMError(f"Ollama error {response.status_code}: {response.text[:500]}")
            for line in response.iter_lines():
                if not line:
                    continue
                piece = json.loads(line).get("message", {}).get("content", "")
                # Hide inline <think> blocks from reasoning models.
                if "<think>" in piece:
                    in_think = True
                if in_think:
                    if "</think>" in piece:
                        in_think = False
                        piece = piece.split("</think>", 1)[1]
                    else:
                        continue
                if piece:
                    yield piece

    def embed(self, texts: list[str]) -> list[list[float]]:
        body = {"model": self.cfg.embed_model, "input": texts, "truncate": True, "keep_alive": self.cfg.keep_alive}
        return self._post("/api/embed", body)["embeddings"]

    def models(self) -> list[str]:
        response = self.http.get("/api/tags")
        response.raise_for_status()
        return [m["name"] for m in response.json().get("models", [])]


class OpenAICompatClient:
    """LM Studio: `lms server start` (default http://localhost:1234/v1).
    On Apple Silicon the MLX builds of Qwen2.5-VL are noticeably faster."""

    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        headers = {"Authorization": f"Bearer {cfg.api_key}"} if cfg.api_key else {}
        self.http = httpx.Client(base_url=cfg.url, timeout=cfg.timeout, headers=headers)

    def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        def call():
            response = self.http.post(path, json=body)
            response.raise_for_status()
            return response.json()

        return _with_retries(call)

    def vision_json(self, system: str, prompt: str, image: bytes, schema: dict[str, Any]) -> str:
        data_url = "data:image/jpeg;base64," + base64.b64encode(image).decode()
        body = {
            "model": self.cfg.vision_model,
            "messages": [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                },
            ],
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "screenshot_entry", "strict": True, "schema": schema},
            },
        }
        return strip_thinking(self._post("/chat/completions", body)["choices"][0]["message"]["content"] or "")

    def chat(self, system: str, prompt: str, stream: bool = False) -> Iterator[str]:
        body = {
            "model": self.cfg.chat_model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            "temperature": max(self.cfg.temperature, 0.3),
            "max_tokens": self.cfg.max_tokens,
            "stream": stream,
        }
        if not stream:
            yield strip_thinking(self._post("/chat/completions", body)["choices"][0]["message"]["content"] or "")
            return
        with self.http.stream("POST", "/chat/completions", json=body) as response:
            if response.status_code >= 400:
                response.read()
                raise LLMError(f"Server error {response.status_code}: {response.text[:500]}")
            for line in response.iter_lines():
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                choices = json.loads(payload).get("choices") or [{}]
                piece = choices[0].get("delta", {}).get("content")
                if piece:
                    yield piece

    def embed(self, texts: list[str]) -> list[list[float]]:
        data = self._post("/embeddings", {"model": self.cfg.embed_model, "input": texts})["data"]
        return [item["embedding"] for item in sorted(data, key=lambda d: d.get("index", 0))]

    def models(self) -> list[str]:
        response = self.http.get("/models")
        response.raise_for_status()
        return [m["id"] for m in response.json().get("data", [])]


def make_llm(cfg: LLMConfig) -> LLM:
    return OllamaClient(cfg) if cfg.provider == "ollama" else OpenAICompatClient(cfg)


def parse_json(text: str) -> dict[str, Any]:
    """Parse model output, tolerating code fences or chatter around the object."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        return json.loads(text[start : end + 1])
    raise LLMError(f"Model did not return JSON: {text[:200]!r}")
