import json
from zipfile import ZipFile

import httpx
import pytest
from test_pipeline import FakeModel

from translate_books import cli
from translate_books.model import Cache, OpenAICompatible, TranslationError


def completion(text="译文", reason="stop"):
    return {"choices": [{"message": {"content": text}, "finish_reason": reason}]}


@pytest.mark.parametrize(
    "path,expected",
    [
        ("", "/v1/"),
        ("/v1/", "/v1/"),
        ("/v1/chat/completions", "/v1/"),
        ("/custom/v2", "/custom/v2/"),
    ],
)
def test_normalize(path, expected):
    assert (
        OpenAICompatible.normalize_url("https://example.com" + path)
        == "https://example.com" + expected
    )


@pytest.mark.parametrize(
    "url",
    ["localhost:8080", "ftp://host", "https://key@host", "https://host?key=x", "https://host#x"],
)
def test_bad_urls(url):
    with pytest.raises(TranslationError):
        OpenAICompatible.normalize_url(url)


@pytest.mark.parametrize("field", ["max_tokens", "max_completion_tokens"])
def test_payload_cache_and_credentials(tmp_path, field):
    calls = []

    def handle(request):
        calls.append(request)
        return httpx.Response(200, json=completion())

    cache = Cache(tmp_path / "cache.sqlite3")
    try:
        for host, name, key in [
            ("one", "model", "secret-key"),
            ("one", "model", "new-key"),
            ("two", "model", None),
            ("one", "other-model", None),
        ]:
            model = OpenAICompatible(
                f"https://{host}/v1",
                cache,
                api_key=key,
                token_limit_field=field,
                transport=httpx.MockTransport(handle),
            )
            try:
                model.check_models([name])
                assert model.generate(name, "hello", max_tokens=100) == "译文"
            finally:
                model.close()
        assert len(calls) == 3
        request = calls[0]
        assert request.method == "POST" and request.url.path == "/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer secret-key"
        assert "authorization" not in calls[1].headers
        assert json.loads(request.content) == {
            "model": "model",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": False,
            field: 100,
        }
    finally:
        cache.close()
    assert b"secret-key" not in (tmp_path / "cache.sqlite3").read_bytes()


@pytest.mark.parametrize(
    "response",
    [
        completion("partial", "length"),
        completion("", "stop"),
        completion("text", "tool_calls"),
        {"choices": []},
        [],
        {"choices": [{"finish_reason": "stop", "message": {"content": "text", "refusal": "no"}}]},
    ],
)
def test_invalid_response_not_cached(tmp_path, response):
    cache = Cache(tmp_path / "cache.sqlite3")
    model = OpenAICompatible(
        "https://example.com",
        cache,
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=response)),
    )
    try:
        with pytest.raises(TranslationError):
            model.generate("model", "prompt")
        assert cache.db.execute("SELECT count(*) FROM responses").fetchone()[0] == 0
    finally:
        model.close()
        cache.close()


def test_redacts_key_from_errors(tmp_path):
    cache = Cache(tmp_path / "cache.sqlite3")
    model = OpenAICompatible(
        "https://example.com",
        cache,
        api_key="secret-key",
        transport=httpx.MockTransport(lambda _: httpx.Response(401, text="invalid secret-key")),
    )
    try:
        with pytest.raises(TranslationError) as error:
            model.generate("model", "prompt")
        assert "secret-key" not in str(error.value)
        assert "[REDACTED]" in str(error.value)
    finally:
        model.close()
        cache.close()


def test_cli_compatible_end_to_end(epub_path, tmp_path, monkeypatch):
    fake = FakeModel()
    payloads = []

    def handle(request):
        assert str(request.url) == "https://configured.example/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer test-key"
        data = json.loads(request.content)
        payloads.append(data)
        return httpx.Response(
            200, json=completion(fake.generate(data["model"], data["messages"][0]["content"]))
        )

    class MockAPI(OpenAICompatible):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs, transport=httpx.MockTransport(handle))

    monkeypatch.setattr(cli, "OpenAICompatible", MockAPI)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://configured.example/v1")
    monkeypatch.setenv("CUSTOM_KEY", "test-key")
    args = [
        str(epub_path),
        "--backend",
        "openai",
        "--model",
        "translator",
        "--summary-model",
        "summarizer",
        "--api-key-env",
        "CUSTOM_KEY",
        "--quiet",
    ]
    assert cli.main(args) == 0
    assert {p["model"] for p in payloads if "〖材料〗" in p["messages"][0]["content"]} == {
        "summarizer"
    }
    assert {p["model"] for p in payloads if "〖待翻译文本〗" in p["messages"][0]["content"]} == {
        "translator"
    }
    with ZipFile(tmp_path / "book.zh-Hans.epub") as archive:
        assert "译文" in archive.read("OPS/text/chapter1.xhtml").decode()
    count = len(payloads)
    assert cli.main(args + ["--force"]) == 0
    assert len(payloads) == count


@pytest.mark.parametrize(
    "args",
    [
        ["--backend", "openai"],
        ["--base-url", "https://example.com"],
        ["--backend", "openai", "--base-url", "https://example.com"],
    ],
)
def test_cli_requires_explicit_backend_model_and_endpoint(epub_path, monkeypatch, args):
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    assert cli.main([str(epub_path), *args]) == 1
