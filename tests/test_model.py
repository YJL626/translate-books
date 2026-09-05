import json

import httpx
import pytest

from translate_books.model import DEFAULT_MODEL, Cache, Ollama, TranslationError


def test_cache_reuse_and_model_digest_invalidation(tmp_path):
    requests = []
    digest = ["v1"]

    def handle(request):
        if request.url.path == "/api/tags":
            return httpx.Response(
                200, json={"models": [{"name": DEFAULT_MODEL, "digest": digest[0]}]}
            )
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={"response": "译文", "done": True, "done_reason": "stop"})

    path = tmp_path / "cache.sqlite3"
    for version in ["v1", "v1", "v2"]:
        digest[0] = version
        cache = Cache(path)
        client = Ollama("http://localhost:11434", cache, transport=httpx.MockTransport(handle))
        client.check_models([DEFAULT_MODEL])
        assert client.generate(DEFAULT_MODEL, "翻译") == "译文"
        client.close()
        cache.close()
    assert len(requests) == 2
    assert requests[0]["model"] == DEFAULT_MODEL
    assert requests[0]["options"]["num_ctx"] == 16384


@pytest.mark.parametrize(
    "result",
    [
        {"response": "partial", "done": True, "done_reason": "length"},
        {"response": "", "done": True},
        {"response": "partial", "done": False},
        {"error": "model failed"},
    ],
)
def test_invalid_result_is_not_cached(tmp_path, result):
    cache = Cache(tmp_path / "cache.sqlite3")
    client = Ollama(
        "http://localhost:11434",
        cache,
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=result)),
    )
    try:
        with pytest.raises(TranslationError):
            client.generate(DEFAULT_MODEL, "翻译")
        assert cache.db.execute("SELECT count(*) FROM responses").fetchone()[0] == 0
    finally:
        client.close()
        cache.close()


def test_retry_transient_http_failure(tmp_path, monkeypatch):
    calls = []

    def handle(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(503, text="busy")
        return httpx.Response(200, json={"response": "译文", "done": True})

    monkeypatch.setattr("translate_books.model.time.sleep", lambda _: None)
    cache = Cache(tmp_path / "cache.sqlite3")
    client = Ollama("localhost:11434", cache, transport=httpx.MockTransport(handle))
    try:
        assert client.generate(DEFAULT_MODEL, "翻译") == "译文"
        assert len(calls) == 2
    finally:
        client.close()
        cache.close()


def test_missing_model_and_context_limit(tmp_path):
    cache = Cache(tmp_path / "cache.sqlite3")
    client = Ollama(
        "localhost:11434",
        cache,
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json={"models": []})),
    )
    try:
        with pytest.raises(TranslationError, match="ollama pull"):
            client.check_models([DEFAULT_MODEL])
        with pytest.raises(TranslationError, match="上下文预算"):
            client.generate(DEFAULT_MODEL, "字" * 10000)
    finally:
        client.close()
        cache.close()
