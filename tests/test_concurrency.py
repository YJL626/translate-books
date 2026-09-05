import json
import threading
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

import httpx
import pytest
from lxml import etree
from test_pipeline import FakeModel

from translate_books import cli
from translate_books.epub import Epub, Unit
from translate_books.model import DEFAULT_MODEL, Cache, Ollama
from translate_books.pipeline import BookContext, Pipeline, Settings
from translate_books.progress import Progress


@pytest.mark.parametrize(
    "backend,explicit,expected",
    [("ollama", None, 4), ("openai", None, 2), ("ollama", 2, 2), ("openai", 4, 4)],
)
def test_backend_default_workers(epub_path, backend, explicit, expected):
    argv = [str(epub_path), "--backend", backend, "--dry-run"]
    if backend == "openai":
        argv += ["--base-url", "https://example.com/v1", "--model", "model"]
    if explicit is not None:
        argv += ["--workers", str(explicit)]
    args = cli.parser().parse_args(argv)
    cli.run(args)
    assert args.workers == expected
    assert Settings().workers == 4


@pytest.mark.parametrize("workers", [2, 4])
def test_workers_overlap_and_preserve_document_order(epub_path, workers):
    class ConcurrentModel(FakeModel):
        def __init__(self):
            super().__init__()
            self.lock = threading.Lock()
            self.barrier = threading.Barrier(workers)
            self.active = self.peak = self.started = 0

        def generate(self, *args, **kwargs):
            with self.lock:
                self.active += 1
                self.started += 1
                index = self.started
                self.peak = max(self.peak, self.active)
            try:
                if index <= workers:
                    self.barrier.wait(timeout=5)
                return super().generate(*args, **kwargs)
            finally:
                with self.lock:
                    self.active -= 1

    model = ConcurrentModel()
    book = Epub(epub_path)
    messages = []
    before = [
        [(node.tag, dict(node.attrib)) for node in document.tree.getroot().iter()]
        for document in book.documents
    ]
    try:
        Pipeline(model, Settings(workers=workers), messages.append).translate(
            book, BookContext("全书摘要", {})
        )
        assert model.peak == workers
        assert [
            [(node.tag, dict(node.attrib)) for node in document.tree.getroot().iter()]
            for document in book.documents
        ] == before
        assert book.chapters[0].tree.find(".//{*}h1").text == "译文"
        assert book.chapters[1].tree.find(".//{*}h1").text == "译文"
        assert f"并发 {workers}" in messages[-1]
        assert "100.0%" in messages[-1]
    finally:
        book.close()


def test_applying_worker_result_does_not_overwrite_another_units_tail():
    original = etree.fromstring(b"<p>Hello<b>world</b></p>")
    original.tail = "Original tail"
    translated = deepcopy(original)
    translated.text = "你好"
    original.tail = "已经翻译的尾部文本"
    Pipeline._apply_unit(Unit(original), Unit(translated))
    assert original.text == "你好"
    assert original.tail == "已经翻译的尾部文本"


def test_shared_cache_and_thread_local_request_counts(tmp_path):
    barrier = threading.Barrier(2)
    counter_lock = threading.Lock()
    posts = 0

    def handle(request):
        nonlocal posts
        with counter_lock:
            posts += 1
            number = posts
        if number <= 2:
            barrier.wait(timeout=5)
        prompt = json.loads(request.content)["prompt"]
        return httpx.Response(200, json={"response": "译文：" + prompt, "done": True})

    cache = Cache(tmp_path / "parallel.sqlite3")
    model = Ollama("localhost:11434", cache, transport=httpx.MockTransport(handle))

    def translate(index):
        before = model.thread_requests_made
        answer = model.generate(DEFAULT_MODEL, f"Text {index}")
        assert model.generate(DEFAULT_MODEL, f"Text {index}") == answer
        return model.thread_requests_made - before

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            assert list(pool.map(translate, range(20))) == [1] * 20
        assert model.requests_made == 20
        assert model.cache_hits == 20
        assert cache.db.execute("SELECT count(*) FROM responses").fetchone()[0] == 20
    finally:
        model.close()
        cache.close()


def test_parallel_eta_accounts_for_worker_count_and_last_task():
    progress = Progress(10, 1000, workers=2)
    for _ in range(3):
        progress.complete(100, 10, cached=False)
    assert progress.remaining_seconds == pytest.approx(35)
    for _ in range(6):
        progress.complete(100, 10, cached=False)
    assert progress.remaining_seconds == pytest.approx(10)


@pytest.mark.parametrize("workers", ["0", "-1", "17"])
def test_invalid_workers_rejected(epub_path, workers):
    assert cli.main([str(epub_path), "--workers", workers]) == 1
