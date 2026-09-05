import json
import re
from zipfile import ZipFile

import httpx
import pytest
from conftest import CHAPTER_TWO, make_epub
from lxml import etree

from translate_books import cli
from translate_books.epub import Epub, Unit, visible_text
from translate_books.model import DEFAULT_MODEL, Ollama, TranslationError
from translate_books.pipeline import Pipeline, Settings, split_text


class FakeModel:
    num_ctx = 16384

    def __init__(self, malformed=False):
        self.calls = []
        self.malformed = malformed

    def generate(self, model, prompt, **kwargs):
        self.calls.append(prompt)
        if "〖材料〗" in prompt:
            return "内容涉及平面设计。Point → 点；line → 线；contrast → 对比。"
        source = prompt.split("〖待翻译文本〗\n", 1)[1].split("\n务必", 1)[0]
        if source.startswith("<"):
            if self.malformed:
                return "<segment>标签丢失</segment>"
            tree = etree.fromstring(source.encode())
            for node in tree.iter():
                if node.text and any(c.isalpha() for c in node.text):
                    node.text = "译文"
                if node.tail and any(c.isalpha() for c in node.tail):
                    node.tail = "译文"
            return etree.tostring(tree, encoding="unicode")
        return "译文"


@pytest.mark.parametrize("limit", [1, 10, 30, 100])
def test_chunking_preserves_every_character(limit):
    source = " A sentence with words.\n\n中文句子！下一句话。" + "长" * 150 + "   end\n"
    chunks = split_text(source, limit)
    assert "".join(chunks) == source
    assert all(len(chunk) <= limit for chunk in chunks)


@pytest.mark.parametrize("malformed", [False, True])
def test_inline_translation_preserves_structure_and_protected_content(malformed):
    node = etree.fromstring(
        b'<p class="text"> Hello <strong id="ref">world</strong>! '
        b'<code>print(1)</code> End <img src="image.png"/> Image.</p>'
    )
    before = [(item.tag, dict(item.attrib)) for item in node.iter()]
    model = FakeModel(malformed)
    pipeline = Pipeline(model, Settings(), lambda _: None)
    pipeline.translate_unit(Unit(node), "测试上下文")
    assert [(item.tag, dict(item.attrib)) for item in node.iter()] == before
    assert node.find("code").text == "print(1)"
    assert node.find("img").get("src") == "image.png"
    assert "Hello" not in visible_text(node)
    assert node.text.startswith(" ") and node.text.endswith(" ")
    assert all("测试上下文" in prompt for prompt in model.calls)
    if malformed:
        assert len(model.calls) > 2


def test_invalid_proxy_never_partially_applied():
    node = etree.fromstring(b"<p>Hello <b>world</b> tail</p>")
    pipeline = Pipeline(FakeModel(), Settings())
    proxy, originals = pipeline._proxy(node)
    before = etree.tostring(node)
    with pytest.raises(ValueError):
        pipeline._apply_proxy(proxy, originals, '<segment id="0">你好<b id="1"/> 尾巴</segment>')
    assert etree.tostring(node) == before


def test_summary_covers_late_text_before_translation(tmp_path):
    long_text = ("A design sentence. " * 200) + "ZEBRA_FINAL_DISCOVERY"
    source = CHAPTER_TWO.replace("Final idea.", long_text)
    book = Epub(make_epub(tmp_path / "long.epub", {"OPS/text/chapter 2.xhtml": source}))
    model = FakeModel()
    pipeline = Pipeline(model, Settings(summary_chunk_chars=500), lambda _: None)
    try:
        context = pipeline.summarize(book)
        assert not any("〖待翻译文本〗" in call for call in model.calls)
        assert any("ZEBRA_FINAL_DISCOVERY" in call for call in model.calls)
        summary_count = len(model.calls)
        pipeline.translate(book, context)
        assert all("全书摘要与术语" in call for call in model.calls[summary_count:])
        assert len(context.chapters) == 2
    finally:
        book.close()


def test_final_overview_represents_every_chapter(epub_path):
    book = Epub(epub_path)
    model = FakeModel()
    pipeline = Pipeline(model, Settings(), lambda _: None)
    try:
        chapters = {
            book.chapters[0].path: "FIRST_CHAPTER " + "长摘要。" * 2000,
            book.chapters[1].path: "LAST_CHAPTER " + "长摘要。" * 2000,
        }
        pipeline._overview(book, chapters, 2500)
        prompt = model.calls[-1]
        assert "FIRST_CHAPTER" in prompt and "LAST_CHAPTER" in prompt
        assert "Visual Language" in prompt and "Balance and Contrast" in prompt
        assert len(prompt.encode()) + 1600 + 128 <= model.num_ctx
    finally:
        book.close()


def test_cli_dry_run_has_no_side_effects(epub_path, tmp_path, capsys):
    assert cli.main([str(epub_path), "--dry-run"]) == 0
    assert set(tmp_path.iterdir()) == {epub_path}
    assert "chapter1.xhtml" in capsys.readouterr().out


def test_cli_end_to_end_summary_cache_and_archive(epub_path, tmp_path, monkeypatch):
    fake = FakeModel()
    payloads = []

    def handle(request):
        if request.url.path == "/api/tags":
            return httpx.Response(
                200, json={"models": [{"name": DEFAULT_MODEL, "digest": "digest"}]}
            )
        data = json.loads(request.content)
        payloads.append(data)
        answer = fake.generate(data["model"], data["prompt"])
        return httpx.Response(200, json={"response": answer, "done": True, "done_reason": "stop"})

    def factory(*args, **kwargs):
        return Ollama(*args, **kwargs, transport=httpx.MockTransport(handle))

    monkeypatch.setattr(cli, "Ollama", factory)
    args = [str(epub_path), "--quiet"]
    assert cli.main(args + ["--summary-only"]) == 0
    assert not (tmp_path / "book.zh-Hans.epub").exists()
    summary_calls = len(payloads)
    assert summary_calls >= 3
    assert (tmp_path / "book.zh-Hans.summary.md").exists()
    assert cli.main(args) == 0
    assert all("〖待翻译文本〗" in p["prompt"] for p in payloads[summary_calls:])
    with ZipFile(tmp_path / "book.zh-Hans.epub") as archive:
        html = archive.read("OPS/text/chapter1.xhtml").decode()
        assert "译文" in html and 'id="intro"' in html
        assert "grid-template-columns: 1fr 1fr" in html
        assert "Keep this original." in html
        assert "Visual Language" not in archive.read("OPS/toc.ncx").decode()
        assert "Visual Language" not in archive.read("OPS/nav.xhtml").decode()
    complete_calls = len(payloads)
    assert cli.main(args + ["--force"]) == 0
    assert len(payloads) == complete_calls
    assert cli.main(args + ["--force", "--workers", "2"]) == 0
    assert len(payloads) == complete_calls
    assert cli.main(args) == 1


@pytest.mark.parametrize("separate_summary", [False, True])
def test_custom_models_reach_the_correct_pipeline_stages(epub_path, monkeypatch, separate_summary):
    translation_model = "custom-translator:latest"
    summary_model = "custom-summary:latest" if separate_summary else translation_model
    payloads = []
    fake = FakeModel()

    def handle(request):
        if request.url.path == "/api/tags":
            return httpx.Response(
                200,
                json={
                    "models": [
                        {"name": name, "digest": name}
                        for name in {translation_model, summary_model}
                    ]
                },
            )
        data = json.loads(request.content)
        payloads.append(data)
        return httpx.Response(
            200, json={"response": fake.generate(data["model"], data["prompt"]), "done": True}
        )

    def factory(*args, **kwargs):
        return Ollama(*args, **kwargs, transport=httpx.MockTransport(handle))

    monkeypatch.setattr(cli, "Ollama", factory)
    args = [str(epub_path), "--model", translation_model, "--quiet"]
    if separate_summary:
        args.extend(["--summary-model", summary_model])
    assert cli.main(args) == 0
    summaries = [p for p in payloads if "〖材料〗" in p["prompt"]]
    translations = [p for p in payloads if "〖待翻译文本〗" in p["prompt"]]
    assert summaries and translations
    assert {p["model"] for p in summaries} == {summary_model}
    assert {p["model"] for p in translations} == {translation_model}


def test_interruption_never_publishes_partial_epub(epub_path, tmp_path, monkeypatch):
    fake = FakeModel()

    class InterruptedModel(FakeModel):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def check_models(self, models):
            pass

        def close(self):
            pass

        def generate(self, model, prompt, **kwargs):
            if "〖待翻译文本〗" in prompt:
                raise TranslationError("模拟模型中断")
            return fake.generate(model, prompt)

    monkeypatch.setattr(cli, "Ollama", InterruptedModel)
    assert cli.main([str(epub_path), "--quiet"]) == 1
    assert not (tmp_path / "book.zh-Hans.epub").exists()
    assert (tmp_path / "book.zh-Hans.summary.md").exists()


@pytest.mark.parametrize("option", ["--output", "--summary-output", "--cache"])
def test_reject_input_collision(epub_path, option):
    before = epub_path.read_bytes()
    assert cli.main([str(epub_path), option, str(epub_path)]) == 1
    assert epub_path.read_bytes() == before


def test_custom_language_requires_code(epub_path):
    assert cli.main([str(epub_path), "--dry-run", "-t", "葡萄牙语"]) == 1
    assert cli.main([str(epub_path), "--dry-run", "-t", "葡萄牙语", "--lang-code", "pt"]) == 0


def test_translation_chunks_use_bounded_context():
    model = FakeModel()
    pipeline = Pipeline(model, Settings(chunk_chars=500), lambda _: None)
    pipeline.translate_text("Text " * 400, "背景" * 400)
    assert len(model.calls) > 1
    assert all(len(prompt.encode()) + 4224 <= model.num_ctx for prompt in model.calls)
    assert all(re.search("背景", prompt) for prompt in model.calls)
