from __future__ import annotations

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

CONTAINER = """<?xml version="1.0"?>
<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container" version="1.0">
<rootfiles><rootfile full-path="OPS/book.opf"
media-type="application/oebps-package+xml"/></rootfiles></container>"""
PACKAGE = """<package xmlns="http://www.idpf.org/2007/opf" version="3.0"
unique-identifier="book-id"><metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
<dc:identifier id="book-id">urn:uuid:7db2144b-46c6-4391-9a3e-9f3bb9f02140</dc:identifier>
<dc:title>Design Notes</dc:title><dc:language>en</dc:language>
<meta property="dcterms:modified">2026-09-05T00:00:00Z</meta></metadata>
<manifest>
<item id="two" href="text/chapter%202.xhtml" media-type="application/xhtml+xml"/>
<item id="one" href="text/chapter1.xhtml" media-type="application/xhtml+xml"/>
<item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>
<item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml"/>
<item id="css" href="style.css" media-type="text/css"/>
<item id="img" href="cover.svg" media-type="image/svg+xml"/>
</manifest><spine toc="ncx"><itemref idref="one"/><itemref idref="two"/></spine>
</package>"""
CHAPTER_ONE = """<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="en">
<head><title>Design Notes</title><link rel="stylesheet" href="../style.css"/></head>
<body><h1 id="intro">Visual Language</h1>
<p class="text">Point, <strong>line</strong>, and plane are the building blocks of design.</p>
<p>A <a href="chapter%202.xhtml#balance">balanced composition</a> guides the eye.</p>
<p>Use <code>grid-template-columns: 1fr 1fr</code> to create columns.</p>
<p translate="no">Keep this original.</p>
<p><img src="../cover.svg" alt="Cover"/> A simple diagram.</p>
<pre>console.log('preserved')</pre><script>const untouched = true;</script>
</body></html>"""
CHAPTER_TWO = """<html xmlns="http://www.w3.org/1999/xhtml" lang="en">
<head><title>Balance</title></head><body><h1 id="balance">Balance and Contrast</h1>
<p>Contrast creates hierarchy. Rhythm helps readers follow an idea.</p>
<div>Text outside paragraphs. <span>Small details matter.</span><p>Final idea.</p> Tail text.</div>
</body></html>"""
NAV = """<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops">
<head><title>Contents</title></head><body><nav epub:type="toc"><h1>Contents</h1><ol>
<li><a href="text/chapter1.xhtml#intro">Visual Language</a></li>
<li><a href="text/chapter%202.xhtml#balance">Balance and Contrast</a></li>
</ol></nav></body></html>"""
NCX = """<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/" version="2005-1">
<head><meta name="dtb:uid" content="urn:uuid:7db2144b-46c6-4391-9a3e-9f3bb9f02140"/></head>
<docTitle><text>Design Notes</text></docTitle><navMap>
<navPoint id="one" playOrder="1"><navLabel><text>Visual Language</text></navLabel>
<content src="text/chapter1.xhtml#intro"/></navPoint></navMap></ncx>"""


def make_epub(path: Path, overrides: dict[str, str | bytes] | None = None) -> Path:
    files = {
        "mimetype": "application/epub+zip",
        "META-INF/container.xml": CONTAINER,
        "OPS/book.opf": PACKAGE,
        "OPS/text/chapter1.xhtml": CHAPTER_ONE,
        "OPS/text/chapter 2.xhtml": CHAPTER_TWO,
        "OPS/nav.xhtml": NAV,
        "OPS/toc.ncx": NCX,
        "OPS/style.css": "p { color: #222; } strong { font-weight: bold; }",
        "OPS/cover.svg": '<svg xmlns="http://www.w3.org/2000/svg" width="40" height="40">'
        '<circle cx="20" cy="20" r="15" fill="blue"/></svg>',
    }
    files.update(overrides or {})
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    return path


@pytest.fixture
def epub_path(tmp_path):
    return make_epub(tmp_path / "book.epub")
