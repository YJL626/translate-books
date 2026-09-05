from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

from .epub import Epub
from .model import DEFAULT_MODEL, Cache, Ollama, OpenAICompatible, TranslationError
from .pipeline import Pipeline, Settings

LANGUAGES = {
    "zh": ("简体中文", "zh-Hans"),
    "zh-cn": ("简体中文", "zh-Hans"),
    "zh-hans": ("简体中文", "zh-Hans"),
    "简体中文": ("简体中文", "zh-Hans"),
    "中文": ("简体中文", "zh-Hans"),
    "zh-tw": ("繁体中文", "zh-Hant"),
    "zh-hant": ("繁体中文", "zh-Hant"),
    "繁体中文": ("繁体中文", "zh-Hant"),
    "en": ("英语", "en"),
    "英语": ("英语", "en"),
    "ja": ("日语", "ja"),
    "ko": ("韩语", "ko"),
    "fr": ("法语", "fr"),
    "de": ("德语", "de"),
    "es": ("西班牙语", "es"),
    "ru": ("俄语", "ru"),
}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        prog="translate-books",
        description="使用 Ollama 或 OpenAI 兼容接口：先概括全书，再结合上下文翻译 EPUB。",
    )
    result.add_argument("input", type=Path, help="输入 EPUB 路径")
    result.add_argument("-o", "--output", type=Path, help="输出路径；默认 原名.zh-Hans.epub")
    result.add_argument(
        "-t", "--target", default="zh-Hans", help="目标语言，默认 zh-Hans（简体中文）"
    )
    result.add_argument("--lang-code", help="自定义目标语言的 BCP 47 代码，如 pt-BR")
    result.add_argument("--model", help=f"翻译模型；Ollama 默认 {DEFAULT_MODEL}，兼容接口需指定")
    result.add_argument("--summary-model", help="摘要模型；默认与翻译模型相同")
    result.add_argument(
        "--backend", choices=["ollama", "openai"], default="ollama", help="接口类型，默认 ollama"
    )
    result.add_argument("--base-url", help="OpenAI 兼容接口地址，也可设置 OPENAI_BASE_URL")
    result.add_argument(
        "--api-key-env", default="OPENAI_API_KEY", help="API Key 的环境变量名，默认 OPENAI_API_KEY"
    )
    result.add_argument(
        "--openai-token-limit",
        choices=["max_tokens", "max_completion_tokens"],
        default="max_tokens",
        help="兼容接口的输出 token 参数名，默认 max_tokens",
    )
    result.add_argument(
        "--host",
        default=os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
        help="Ollama 地址，也可设置 OLLAMA_HOST",
    )
    result.add_argument("--num-ctx", type=int, default=16384, help="上下文窗口，默认 16384 tokens")
    result.add_argument("--chunk-chars", type=int, default=1200, help="翻译块字符上限，默认 1200")
    result.add_argument(
        "--summary-chunk-chars", type=int, default=2500, help="摘要输入块字符上限，默认 2500"
    )
    result.add_argument(
        "--summary-chars", type=int, default=1000, help="全书摘要字符上限，默认 1000"
    )
    result.add_argument("--timeout", type=float, default=300, help="单次请求超时秒数，默认 300")
    result.add_argument("--retries", type=int, default=2, help="网络或服务端错误重试次数，默认 2")
    result.add_argument(
        "--workers", type=int, help="同时翻译的段落数；Ollama 默认 4，兼容接口默认 2；设为 1 串行"
    )
    result.add_argument("--cache", type=Path, help="SQLite 缓存路径，默认与输出同目录")
    result.add_argument("--summary-output", type=Path, help="摘要 Markdown 路径，默认与输出同名")
    result.add_argument("--summary-only", action="store_true", help="只生成全书和章节摘要")
    result.add_argument("--dry-run", action="store_true", help="仅检查 EPUB 和统计文本，不调用模型")
    result.add_argument("--force", action="store_true", help="允许覆盖已有译本（仍禁止覆盖原书）")
    result.add_argument("--quiet", action="store_true", help="隐藏逐段进度，只显示结果和错误")
    return result


def run(args: argparse.Namespace) -> None:
    if args.workers is None:
        args.workers = 4 if args.backend == "ollama" else 2
    model_name = args.model or DEFAULT_MODEL
    base_url = None
    api_key = None
    if args.backend == "openai":
        base_url = args.base_url or os.environ.get("OPENAI_BASE_URL")
        if not base_url or not args.model:
            raise TranslationError(
                "使用兼容接口请指定 --model，并设置 --base-url 或 OPENAI_BASE_URL。"
            )
        base_url = OpenAICompatible.normalize_url(base_url)
        api_key = os.environ.get(args.api_key_env)
        if args.api_key_env != "OPENAI_API_KEY" and not api_key:
            raise TranslationError(f"未设置 API Key 环境变量：{args.api_key_env}")
    elif args.base_url:
        raise TranslationError("--base-url 需要同时指定 --backend openai；Ollama 地址使用 --host。")
    if args.num_ctx < 8192:
        raise TranslationError("--num-ctx 至少为 8192。")
    if args.chunk_chars < 64 or args.summary_chunk_chars < 256 or args.summary_chars < 128:
        raise TranslationError("分块参数过小：翻译至少 64、摘要输入至少 256、摘要长度至少 128。")
    if args.timeout <= 0 or args.retries < 0:
        raise TranslationError("--timeout 必须为正数，--retries 不能为负数。")
    if not 1 <= args.workers <= 16:
        raise TranslationError("--workers 必须在 1 到 16 之间。")
    target, code = LANGUAGES.get(args.target.lower(), (args.target, args.lang_code))
    code = args.lang_code or code
    if not code or not re.fullmatch(r"[a-zA-Z]{2,8}(?:-[a-zA-Z0-9]{1,8})*", code):
        raise TranslationError(
            "自定义目标语言请同时指定 --lang-code，例如 -t 葡萄牙语 --lang-code pt。"
        )
    output = args.output or args.input.with_name(f"{args.input.stem}.{code}.epub")
    summary_path = args.summary_output or output.with_suffix(".summary.md")
    cache_path = args.cache or output.with_suffix(".translate-cache.sqlite3")
    all_paths = [args.input, output, summary_path, cache_path]
    if len({path.resolve() for path in all_paths}) != len(all_paths):
        raise TranslationError("原书、译本、摘要和缓存路径必须互不相同。")
    if not args.dry_run and not args.summary_only and output.exists() and not args.force:
        raise TranslationError(f"输出文件已存在：{output}；使用 --force 明确覆盖。")
    book = Epub(args.input)
    try:
        print(f"书名：{book.title}", file=sys.stderr)
        print(
            f"源语言：{book.language} → {target}；正文文件 {len(book.chapters)}；"
            f"待翻译文档 {len(book.documents)}；"
            f"段落 {sum(len(doc.units) for doc in book.documents)}；"
            f"正文字符 {sum(len(doc.text) for doc in book.chapters):,}",
            file=sys.stderr,
        )
        if args.dry_run:
            for chapter in book.chapters:
                print(f"{chapter.path}\t{len(chapter.text):,} 字符")
            return
        settings = Settings(
            model=model_name,
            summary_model=args.summary_model or model_name,
            target=target,
            language=code,
            chunk_chars=args.chunk_chars,
            summary_chunk_chars=args.summary_chunk_chars,
            summary_chars=args.summary_chars,
            workers=args.workers,
        )
        cache = Cache(cache_path)
        options = dict(num_ctx=args.num_ctx, timeout=args.timeout, retries=args.retries)
        if args.backend == "openai":
            model = OpenAICompatible(
                base_url,
                cache,
                api_key=api_key,
                token_limit_field=args.openai_token_limit,
                **options,
            )
        else:
            model = Ollama(args.host, cache, **options)
        try:
            required = [settings.summary_model]
            if not args.summary_only:
                required.append(settings.model)
            model.check_models(required)

            def log(message: str) -> None:
                if not args.quiet:
                    print(message, file=sys.stderr, flush=True)

            pipeline = Pipeline(model, settings, log)
            context = pipeline.summarize(book)
            pipeline.write_summary(book, context, summary_path)
            print(f"摘要已保存：{summary_path}", file=sys.stderr)
            if not args.summary_only:
                pipeline.translate(book, context)
                book.write(output, book.replacements(code), force=args.force)
                print(f"译本已保存：{output}")
            else:
                print(summary_path)
        finally:
            model.close()
            cache.close()
    finally:
        book.close()


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        run(args)
        return 0
    except KeyboardInterrupt:
        print("\n已中断。已完成的模型请求保存在缓存中，重跑同一命令即可继续。", file=sys.stderr)
        return 130
    except (TranslationError, OSError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1
