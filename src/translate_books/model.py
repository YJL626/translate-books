from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path

import httpx

DEFAULT_MODEL = "hf.co/tencent/Hy-MT2-7B-GGUF:Q8_0"


class TranslationError(Exception):
    """An actionable error safe to show in the CLI."""


class Cache:
    """Commit each successful request, so interrupted books can resume."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute("CREATE TABLE IF NOT EXISTS responses (key TEXT PRIMARY KEY, value TEXT)")

    def get(self, key: str) -> str | None:
        with self.lock:
            row = self.db.execute("SELECT value FROM responses WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def put(self, key: str, value: str) -> None:
        with self.lock:
            self.db.execute("INSERT OR REPLACE INTO responses VALUES (?, ?)", (key, value))
            self.db.commit()

    def close(self) -> None:
        with self.lock:
            self.db.close()


class Ollama:
    endpoint = "api/generate"
    service_name = "Ollama"

    def __init__(
        self,
        base_url: str,
        cache: Cache,
        *,
        num_ctx: int = 16384,
        timeout: float = 300,
        retries: int = 2,
        transport: httpx.BaseTransport | None = None,
    ):
        if "://" not in base_url:
            base_url = "http://" + base_url
        self.client = httpx.Client(
            base_url=base_url.rstrip("/") + "/",
            timeout=httpx.Timeout(timeout, connect=10),
            transport=transport,
            trust_env=False,
        )
        self.cache = cache
        self.num_ctx = num_ctx
        self.retries = retries
        self.digests: dict[str, str] = {}
        self.requests_made = 0
        self.cache_hits = 0
        self._stats_lock = threading.Lock()
        self._thread_stats = threading.local()

    @property
    def thread_requests_made(self) -> int:
        return getattr(self._thread_stats, "requests_made", 0)

    def check_models(self, models: list[str]) -> None:
        try:
            response = self.client.get("api/tags")
            response.raise_for_status()
            available = response.json()["models"]
            installed = {m["name"]: m.get("digest", m["name"]) for m in available}
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
            raise TranslationError(
                f"无法连接 Ollama：{self.client.base_url}。请先启动 ollama serve。详情：{exc}"
            ) from exc
        for model in set(models):
            resolved = model if model in installed else model + ":latest"
            if resolved not in installed:
                raise TranslationError(f"本机没有模型 {model}，请先运行：ollama pull {model}")
            self.digests[model] = installed[resolved]

    def _payload(self, model: str, prompt: str, max_tokens: int) -> dict:
        return {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": "10m",
            "options": {
                "num_ctx": self.num_ctx,
                "num_predict": max_tokens,
                "temperature": 0.2,
                "top_p": 0.6,
                "top_k": 20,
                "repeat_penalty": 1.05,
                "seed": 42,
            },
        }

    def _cache_identity(self, model: str, payload: dict) -> dict:
        return {"version": 1, "digest": self.digests.get(model), "request": payload}

    def _result_text(self, result: dict) -> str:
        if not isinstance(result, dict):
            raise TranslationError("模型响应必须是 JSON 对象。")
        if result.get("error"):
            raise TranslationError(str(result["error"]))
        if result.get("done_reason") == "length":
            raise TranslationError("模型输出被截断，请减小分块大小后重试。")
        text = result.get("response", "")
        if not result.get("done") or not isinstance(text, str) or not text.strip():
            raise TranslationError("模型未返回完整的非空结果。")
        return text.strip()

    def _http_detail(self, response: httpx.Response) -> str:
        return response.text[:500]

    def _redact(self, text: str) -> str:
        return text

    def generate(self, model: str, prompt: str, *, max_tokens: int = 4096) -> str:
        # Conservative byte-based upper bound; never silently truncate the book context.
        if len(prompt.encode("utf-8")) + max_tokens + 128 > self.num_ctx:
            raise TranslationError(
                "本次请求超过保守上下文预算，请增大 --num-ctx 或减小 "
                "--chunk-chars / --summary-chunk-chars / --summary-chars。"
            )
        payload = self._payload(model, prompt, max_tokens)
        key = hashlib.sha256(
            json.dumps(
                self._cache_identity(model, payload),
                ensure_ascii=False,
                sort_keys=True,
            ).encode()
        ).hexdigest()
        cached = self.cache.get(key)
        if cached is not None:
            with self._stats_lock:
                self.cache_hits += 1
            return cached
        for attempt in range(self.retries + 1):
            try:
                with self._stats_lock:
                    self.requests_made += 1
                self._thread_stats.requests_made = self.thread_requests_made + 1
                response = self.client.post(self.endpoint, json=payload)
                response.raise_for_status()
                text = self._result_text(response.json())
                self.cache.put(key, text)
                return text
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code < 500 and exc.response.status_code != 429:
                    raise TranslationError(
                        f"{self.service_name} 拒绝请求 ({exc.response.status_code})："
                        f"{self._http_detail(exc.response)}"
                    ) from exc
                error = str(exc)
            except (httpx.HTTPError, ValueError) as exc:
                error = str(exc)
            if attempt < self.retries:
                time.sleep(min(2**attempt, 4))
        raise TranslationError(
            f"{self.service_name} 请求失败，已重试 {self.retries} 次：{self._redact(error)}"
        )

    def close(self) -> None:
        self.client.close()


class OpenAICompatible(Ollama):
    """Chat Completions transport sharing the retry, cache and concurrency machinery."""

    endpoint = "chat/completions"
    service_name = "OpenAI 兼容接口"

    @staticmethod
    def normalize_url(base_url: str) -> str:
        try:
            url = httpx.URL(base_url)
        except httpx.InvalidURL as exc:
            raise TranslationError("--base-url 必须是有效的 HTTP(S) 地址。") from exc
        if (
            url.scheme not in {"http", "https"}
            or not url.host
            or url.userinfo
            or url.query
            or url.fragment
        ):
            raise TranslationError("--base-url 需要 HTTP(S) 地址，不能包含凭据、查询参数或片段。")
        path = url.path.rstrip("/")
        if path.endswith("/chat/completions"):
            path = path[: -len("/chat/completions")]
        if not path:
            path = "/v1"
        return str(url.copy_with(path=path + "/"))

    def __init__(
        self,
        base_url: str,
        cache: Cache,
        *,
        api_key: str | None = None,
        token_limit_field: str = "max_tokens",
        **kwargs,
    ):
        if token_limit_field not in {"max_tokens", "max_completion_tokens"}:
            raise TranslationError("不支持的输出 token 参数。")
        self._api_key = api_key or ""
        self.token_limit_field = token_limit_field
        super().__init__(self.normalize_url(base_url), cache, **kwargs)
        if self._api_key:
            self.client.headers["Authorization"] = f"Bearer {self._api_key}"

    def check_models(self, models: list[str]) -> None:
        # Some compatible gateways do not implement GET /models. Validate on generation.
        if any(not model or not model.strip() for model in models):
            raise TranslationError("OpenAI 兼容接口必须显式指定 --model。")

    def _payload(self, model: str, prompt: str, max_tokens: int) -> dict:
        return {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            self.token_limit_field: max_tokens,
        }

    def _cache_identity(self, model: str, payload: dict) -> dict:
        return {
            "version": 1,
            "backend": "openai",
            "base_url": str(self.client.base_url),
            "num_ctx": self.num_ctx,
            "request": payload,
        }

    def _redact(self, text: str) -> str:
        return text.replace(self._api_key, "[REDACTED]") if self._api_key else text

    def _http_detail(self, response: httpx.Response) -> str:
        return self._redact(response.text)[:500]

    def _result_text(self, result: dict) -> str:
        try:
            if result.get("error"):
                raise TranslationError(self._redact(str(result["error"])))
            choice = result["choices"][0]
            reason = choice.get("finish_reason")
            if reason == "length":
                raise TranslationError("模型输出被截断，请减小分块大小或检查模型输出预算。")
            message = choice["message"]
            if reason != "stop" or message.get("refusal") or message.get("tool_calls"):
                raise TranslationError("接口未返回正常完成的文本译文（拒绝、过滤或工具调用）。")
            content = message.get("content")
            if isinstance(content, list):
                if any(part.get("type") != "text" for part in content):
                    raise TranslationError("接口返回了非文本内容。")
                content = "".join(part["text"] for part in content)
            if not isinstance(content, str) or not content.strip():
                raise TranslationError("接口返回了空译文。")
            return content.strip()
        except (KeyError, IndexError, TypeError, AttributeError) as exc:
            raise TranslationError("响应不是有效的 Chat Completions 格式。") from exc
