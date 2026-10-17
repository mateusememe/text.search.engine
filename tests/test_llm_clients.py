"""The exact HTTP payloads sent to Ollama and to LM Studio (OpenAI-compatible)."""

import json

import httpx
import pytest

from shotvault.config import LLMConfig
from shotvault.llm import LLMUnavailable, OllamaClient, OpenAICompatClient, parse_json, strip_thinking


def mock(client, handler):
    client.http = httpx.Client(base_url=client.cfg.url, transport=httpx.MockTransport(handler))
    return client


def test_ollama_vision_request():
    seen = {}

    def handler(request):
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"message": {"content": '<think>hm</think>{"title": "x"}'}})

    client = mock(OllamaClient(LLMConfig(num_ctx=8192)), handler)
    out = client.vision_json("sys", "prompt", b"\xff\xd8jpeg", {"type": "object"})
    body = seen["body"]
    assert seen["path"] == "/api/chat"
    assert out == '{"title": "x"}'
    assert body["model"] == "qwen2.5vl:7b"
    assert body["format"] == {"type": "object"}
    assert body["options"]["num_ctx"] == 8192
    assert body["messages"][1]["images"][0] == "/9hqcGVn"
    assert "think" not in body


def test_ollama_embed_and_stream():
    def handler(request):
        if request.url.path == "/api/embed":
            return httpx.Response(200, json={"embeddings": [[0.1, 0.2]]})
        lines = [{"message": {"content": "Hel"}}, {"message": {"content": "lo"}}, {"done": True}]
        return httpx.Response(200, content="\n".join(json.dumps(x) for x in lines))

    client = mock(OllamaClient(LLMConfig()), handler)
    assert client.embed(["a"]) == [[0.1, 0.2]]
    assert "".join(client.chat("s", "p", stream=True)) == "Hello"


def test_openai_compatible_request():
    seen = {}

    def handler(request):
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"title": "y"}'}}]})

    client = mock(OpenAICompatClient(LLMConfig(provider="openai", vision_model="qwen2.5-vl-7b-instruct")), handler)
    assert client.vision_json("sys", "prompt", b"img", {"type": "object"}) == '{"title": "y"}'
    body = seen["body"]
    assert seen["path"] == "/v1/chat/completions"
    image_part = body["messages"][1]["content"][1]
    assert image_part["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["strict"] is True


def test_missing_model_stops_the_run():
    client = mock(OllamaClient(LLMConfig()), lambda r: httpx.Response(404, json={"error": "model not found"}))
    with pytest.raises(LLMUnavailable):
        client.vision_json("s", "p", b"x", {})


def test_parse_json_tolerates_fences():
    assert parse_json('Sure!\n```json\n{"a": 1}\n```') == {"a": 1}
    assert strip_thinking("<think>\nlong\n</think>\nanswer") == "answer"
