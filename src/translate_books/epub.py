from __future__ import annotations

import os
import posixpath
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlsplit
from zipfile import ZIP_STORED, BadZipFile, ZipFile, ZipInfo

from lxml import etree

from .model import TranslationError

XHTML = "http://www.w3.org/1999/xhtml"
XML_LANG = "{http://www.w3.org/XML/1998/namespace}lang"
DC = "http://purl.org/dc/elements/1.1/"
PROTECTED = {"script", "style", "pre", "code", "kbd", "samp", "svg", "math"}
BLOCKS = {
    "address",
    "article",
    "aside",
    "blockquote",
    "body",
    "caption",
    "dd",
    "div",
    "dl",
    "dt",
    "figcaption",
    "figure",
    "footer",
    "form",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "header",
    "hr",
    "li",
    "main",
    "nav",
    "ol",
    "p",
    "section",
    "table",
    "tbody",
    "td",
    "th",
    "thead",
    "tr",
    "ul",
}


def local_name(node: etree._Element) -> str:
    return etree.QName(node).localname if isinstance(node.tag, str) else ""


def protected(node: etree._Element) -> bool:
    return (
        not isinstance(node.tag, str)
        or local_name(node) in PROTECTED
        or etree.QName(node).namespace not in (None, XHTML)
        or node.get("translate", "").lower() == "no"
    )


def has_words(text: str | None) -> bool:
    return bool(text) and any(char.isalpha() for char in text)


def slots(node: etree._Element) -> Iterator[tuple[etree._Element, str]]:
    if protected(node):
        return
    if node.text:
        yield node, "text"
    for child in node:
        yield from slots(child)
        if child.tail:
            yield child, "tail"


def visible_text(node: etree._Element) -> str:
    return "".join(getattr(element, field) for element, field in slots(node))


def parse_xml(data: bytes, name: str) -> etree._ElementTree:
    try:
        root = etree.fromstring(
            data,
            parser=etree.XMLParser(resolve_entities=False, no_network=True, recover=False),
        )
        if any(isinstance(node, etree._Entity) for node in root.iter()):
            raise TranslationError(f"{name} 包含未展开的 XML 实体，请先转为标准 XHTML。")
        return root.getroottree()
    except etree.XMLSyntaxError as exc:
        raise TranslationError(f"无法解析 {name}，需要合法的 EPUB/XHTML：{exc}") from exc


def serialize(tree: etree._ElementTree) -> bytes:
    return etree.tostring(tree, encoding="utf-8", xml_declaration=True)


def resolve_path(base: str, href: str) -> str:
    parsed = urlsplit(href)
    if parsed.scheme or parsed.netloc:
        raise TranslationError(f"不支持远程 EPUB 文档：{href}")
    path = posixpath.normpath(posixpath.join(posixpath.dirname(base), unquote(parsed.path)))
    if path.startswith("/") or path == ".." or path.startswith("../"):
        raise TranslationError(f"EPUB 内部路径越界：{href}")
    return path


@dataclass
class Unit:
    node: etree._Element
    field: str | None = None

    @property
    def text(self) -> str:
        return getattr(self.node, self.field) if self.field else visible_text(self.node)


def collect_units(node: etree._Element) -> Iterator[Unit]:
    if protected(node):
        return
    if not any(local_name(child) in BLOCKS for child in node.iterdescendants()):
        if has_words(visible_text(node)):
            yield Unit(node)
        return
    if has_words(node.text):
        yield Unit(node, "text")
    for child in node:
        yield from collect_units(child)
        if has_words(child.tail):
            yield Unit(child, "tail")


@dataclass
class Document:
    path: str
    tree: etree._ElementTree
    units: list[Unit]
    is_nav: bool = False

    @property
    def text(self) -> str:
        body = self.tree.find(".//{*}body")
        if body is None:
            return ""
        return "\n".join(unit.text for unit in collect_units(body))


class Epub:
    def __init__(self, path: Path):
        self.path = path
        try:
            self.archive = ZipFile(path)
        except (BadZipFile, OSError) as exc:
            raise TranslationError(f"无法打开 EPUB：{exc}") from exc
        try:
            self._read()
        except BaseException:
            self.archive.close()
            raise

    def _read(self) -> None:
        names = self.archive.namelist()
        if len(names) != len(set(names)):
            raise TranslationError("EPUB 内含重复文件名，无法安全重打包。")
        if self.read("mimetype").strip() != b"application/epub+zip":
            raise TranslationError("输入文件不是有效 EPUB：mimetype 不正确。")
        container = parse_xml(self.read("META-INF/container.xml"), "container.xml")
        rootfiles = container.findall(".//{*}rootfile")
        if len(rootfiles) != 1:
            raise TranslationError("目前仅支持包含一个 rendition 的 EPUB。")
        self.opf_path = resolve_path("", rootfiles[0].get("full-path", ""))
        self.opf = parse_xml(self.read(self.opf_path), self.opf_path)
        self.title = self.opf.findtext(f".//{{{DC}}}title") or path_title(self.path)
        self.language = self.opf.findtext(f".//{{{DC}}}language") or "unknown"
        manifest = self.opf.find("{*}manifest")
        spine = self.opf.find("{*}spine")
        if manifest is None or spine is None:
            raise TranslationError("EPUB 缺少 manifest 或 spine。")
        items = {item.get("id"): item for item in manifest}
        order = []
        for itemref in spine:
            key = itemref.get("idref")
            if key not in items:
                raise TranslationError(f"spine 引用了不存在的条目：{key}")
            if key not in order:
                order.append(key)
        spine_ids = set(order)
        order.extend(key for key in items if key not in spine_ids)
        self.documents: list[Document] = []
        self.chapters: list[Document] = []
        self.ncx: dict[str, etree._ElementTree] = {}
        seen = set()
        for key in order:
            item = items[key]
            media = item.get("media-type")
            if media not in {"application/xhtml+xml", "text/html", "application/x-dtbncx+xml"}:
                continue
            name = resolve_path(self.opf_path, item.get("href", ""))
            if name in seen:
                continue
            seen.add(name)
            tree = parse_xml(self.read(name), name)
            if media == "application/x-dtbncx+xml":
                self.ncx[name] = tree
                continue
            body = tree.find(".//{*}body")
            if body is None:
                raise TranslationError(f"{name} 缺少 body。")
            units = list(collect_units(body))
            for title in tree.findall(".//{*}head/{*}title"):
                units.extend(collect_units(title))
            document = Document(name, tree, units, "nav" in item.get("properties", "").split())
            self.documents.append(document)
            if key in spine_ids and not document.is_nav and has_words(document.text):
                self.chapters.append(document)
        if not self.chapters:
            raise TranslationError("EPUB 中没有可提取的正文文本；图片扫描版需要先做 OCR。")
        if "META-INF/encryption.xml" in names:
            encryption = parse_xml(self.read("META-INF/encryption.xml"), "encryption.xml")
            for ref in encryption.findall(".//{*}CipherReference"):
                if resolve_path("", ref.get("URI", "")) in seen | {self.opf_path}:
                    raise TranslationError("EPUB 正文已加密，不支持 DRM 加密书籍。")
        if "META-INF/signatures.xml" in names:
            raise TranslationError("暂不支持带数字签名的 EPUB。")

    def read(self, name: str) -> bytes:
        try:
            return self.archive.read(name)
        except (KeyError, BadZipFile, RuntimeError) as exc:
            raise TranslationError(f"无法读取 EPUB 内部文件 {name}：{exc}") from exc

    def replacements(self, language: str) -> dict[str, bytes]:
        result = {}
        for document in self.documents:
            root = document.tree.getroot()
            root.set(XML_LANG, language)
            if "lang" in root.attrib:
                root.set("lang", language)
            for node in root.iterdescendants():
                if not isinstance(node.tag, str) or protected(node):
                    continue
                if any(protected(ancestor) for ancestor in node.iterancestors()):
                    continue
                for attr in (XML_LANG, "lang"):
                    if attr in node.attrib:
                        node.set(attr, language)
            result[document.path] = serialize(document.tree)
        languages = self.opf.findall(f".//{{{DC}}}language")
        if languages:
            languages[0].text = language
            for extra in languages[1:]:
                extra.getparent().remove(extra)
        else:
            metadata = self.opf.find("{*}metadata")
            if metadata is None:
                raise TranslationError("EPUB 缺少 metadata。")
            etree.SubElement(metadata, f"{{{DC}}}language").text = language
        result[self.opf_path] = serialize(self.opf)
        for name, tree in self.ncx.items():
            tree.getroot().set(XML_LANG, language)
            result[name] = serialize(tree)
        return result

    def write(self, output: Path, replacements: dict[str, bytes], *, force: bool = False) -> None:
        if output.resolve() == self.path.resolve():
            raise TranslationError("输出文件不能覆盖原书。")
        if output.exists() and not force:
            raise TranslationError(f"输出文件已存在：{output}；使用 --force 明确覆盖。")
        output.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
        os.close(descriptor)
        try:
            with ZipFile(temporary, "w") as target:
                target.comment = self.archive.comment
                info = ZipInfo("mimetype")
                info.compress_type = ZIP_STORED
                target.writestr(info, b"application/epub+zip")
                for entry in self.archive.infolist():
                    if entry.filename != "mimetype":
                        data = replacements.get(entry.filename)
                        target.writestr(
                            entry, data if data is not None else self.read(entry.filename)
                        )
            with ZipFile(temporary) as check:
                bad = check.testzip()
                if bad:
                    raise TranslationError(f"输出 EPUB 校验失败：{bad}")
            if force:
                os.replace(temporary, output)
            else:
                # Atomic publish without overwriting a file created during a long run.
                os.link(temporary, output)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def close(self) -> None:
        self.archive.close()


def path_title(path: Path) -> str:
    return path.stem
