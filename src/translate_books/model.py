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

    def generate(self, model: str, prompt: str, *, max_tokens: int = 4096) -> str:
        # Conservative byte-based upper bound; never silently truncate the book context.
        if len(prompt.encode("utf-8")) + max_tokens + 128 > self.num_ctx:
            raise TranslationError(
                "本次请求超过保守上下文预算，请增大 --num-ctx 或减小 "
                "--chunk-chars / --summary-chunk-chars / --summary-chars。"
            )
        payload = {
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
        key = hashlib.sha256(
            json.dumps(
                {"version": 1, "digest": self.digests.get(model), "request": payload},
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
                response = self.client.post("api/generate", json=payload)
                response.raise_for_status()
                result = response.json()
                if result.get("error"):
                    raise TranslationError(str(result["error"]))
                if result.get("done_reason") == "length":
                    raise TranslationError("模型输出被截断，请减小分块大小后重试。")
                text = result.get("response", "")
                if not result.get("done") or not isinstance(text, str) or not text.strip():
                    raise TranslationError("模型未返回完整的非空结果。")
                text = text.strip()
                self.cache.put(key, text)
                return text
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code < 500 and exc.response.status_code != 429:
                    raise TranslationError(
                        f"Ollama 拒绝请求 ({exc.response.status_code})：{exc.response.text[:500]}"
                    ) from exc
                error = str(exc)
            except (httpx.HTTPError, ValueError) as exc:
                error = str(exc)
            if attempt < self.retries:
                time.sleep(min(2**attempt, 4))
        raise TranslationError(f"Ollama 请求失败，已重试 {self.retries} 次：{error}")

    def close(self) -> None:
        self.client.close()
