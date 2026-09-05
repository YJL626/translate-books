from zipfile import ZIP_STORED, ZipFile

import pytest
from conftest import CHAPTER_ONE, CONTAINER, PACKAGE, make_epub
from lxml import etree

from translate_books.epub import Epub, Unit, collect_units, resolve_path, visible_text
from translate_books.model import TranslationError


def test_reading_order_and_visible_text(epub_path):
    book = Epub(epub_path)
    try:
        assert [chapter.path for chapter in book.chapters] == [
            "OPS/text/chapter1.xhtml",
            "OPS/text/chapter 2.xhtml",
        ]
        text = book.chapters[0].text
        assert "Point, line, and plane" in text
        assert "console.log" not in text
        assert "grid-template" not in text
        assert "Keep this original" not in text
        assert "const untouched" not in text
        assert "Tail text." in book.chapters[1].text
        assert "Text outside paragraphs." in book.chapters[1].text
        assert len(book.documents) == 3
    finally:
        book.close()


def test_mixed_content_has_no_duplication():
    root = etree.fromstring(
        b"<body>Before<div>Inside<p>Hello <b>world</b>!</p>Tail</div>After</body>"
    )
    assert [unit.text for unit in collect_units(root)] == [
        "Before",
        "Inside",
        "Hello world!",
        "Tail",
        "After",
    ]
    assert Unit(root).text == "BeforeInsideHello world!TailAfter"


def test_archive_preserves_assets_and_links(epub_path, tmp_path):
    book = Epub(epub_path)
    output = tmp_path / "translated.epub"
    book.chapters[0].units[0].node.text = "视觉语言"
    try:
        book.write(output, book.replacements("zh-Hans"))
        with ZipFile(epub_path) as original, ZipFile(output) as translated:
            assert translated.infolist()[0].filename == "mimetype"
            assert translated.infolist()[0].compress_type == ZIP_STORED
            assert translated.infolist()[0].extra == b""
            assert set(original.namelist()) == set(translated.namelist())
            for name in ["OPS/cover.svg", "OPS/style.css"]:
                assert original.read(name) == translated.read(name)
            html = etree.fromstring(translated.read("OPS/text/chapter1.xhtml"))
            assert html.get("{http://www.w3.org/XML/1998/namespace}lang") == "zh-Hans"
            assert html.find(".//{*}a").get("href") == "chapter%202.xhtml#balance"
            assert html.find(".//{*}h1").text == "视觉语言"
            assert visible_text(html.find(".//{*}h1")) == "视觉语言"
            metadata = etree.fromstring(translated.read("OPS/book.opf"))
            assert metadata.findtext(".//{*}language") == "zh-Hans"
            assert translated.testzip() is None
        with pytest.raises(TranslationError, match="已存在"):
            book.write(output, {})
        with pytest.raises(TranslationError, match="原书"):
            book.write(epub_path, {}, force=True)
    finally:
        book.close()


@pytest.mark.parametrize("href", ["../../outside.xhtml", "/outside.xhtml", "https://example.com/a"])
def test_invalid_member_path(href):
    with pytest.raises(TranslationError):
        resolve_path("OPS/book.opf", href)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"OPS/book.opf": PACKAGE.replace('idref="one"', 'idref="missing"')}, "不存在"),
        ({"OPS/text/chapter1.xhtml": "<html>"}, "无法解析"),
        ({"mimetype": "application/zip"}, "mimetype"),
        ({"META-INF/container.xml": CONTAINER.replace("<rootfile ", "<other ")}, "rendition"),
        ({"META-INF/signatures.xml": "<signatures/>"}, "签名"),
        (
            {
                "OPS/text/chapter1.xhtml": '<!DOCTYPE html [<!ENTITY secret SYSTEM "file:///etc/passwd">]>'
                + CHAPTER_ONE.replace("Visual Language", "&secret;")
            },
            "实体",
        ),
        (
            {
                "META-INF/encryption.xml": "<encryption>"
                '<CipherReference URI="OPS/text/chapter1.xhtml"/>'
                "</encryption>"
            },
            "加密",
        ),
    ],
)
def test_invalid_or_unsupported_books(tmp_path, overrides, message):
    with pytest.raises(TranslationError, match=message):
        Epub(make_epub(tmp_path / "bad.epub", overrides))
