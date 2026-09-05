# translate-books

使用本机 Ollama 翻译 EPUB，默认模型为 `hf.co/tencent/Hy-MT2-7B-GGUF:Q8_0`，默认译为简体中文，正文默认双并发。

按 EPUB spine 阅读顺序提取正文，先对全书分块概括、归并章节摘要，再归并全书摘要与术语。**所有摘要完成后才开始翻译**。每段译文都参考全书摘要和对应章节摘要，摘要另存 Markdown。

## 使用

需要安装 mise 和 Ollama。克隆项目后，通过 mise 安装 Python、uv 和依赖：

```bash
git clone https://github.com/YJL626/translate-books.git
cd translate-books
mise trust
mise install
mise run install

# 如果 Ollama 尚未启动，另开终端运行
ollama serve
# 如果本机尚未安装模型
ollama pull hf.co/tencent/Hy-MT2-7B-GGUF:Q8_0

mise run translate -- "book.epub"
mise run translate -- "book.epub" -o "译本.epub"

# 默认两个段落同时翻译，也可显式指定
mise run translate -- "book.epub" --workers 2
# 如需串行
mise run translate -- "book.epub" --workers 1
```

需要在任意目录直接调用时，可以将虚拟环境中的命令加入 PATH。Linux 下例如：

```bash
sudo ln -s "$PWD/.venv/bin/translate-books" /usr/local/bin/translate-books
translate-books "book.epub" -o "译本.epub"
```

软链接依赖项目目录和 `.venv`，请保留它们。输入输出的相对路径以调用目录为准。

也可使用 `mise exec -- uv run translate-books ...`。命令行选项可通过 `mise run translate -- --help` 查看。

## 指定其他模型

支持任意已在所配置 Ollama 服务中安装、可通过 Generate API 生成文本的模型。使用 `--model` 更换翻译模型；未指定 `--summary-model` 时，摘要也使用该模型。摘要能力和翻译质量取决于所选模型。

```bash
# 查看已安装模型
ollama list

# 摘要和翻译都使用另一个模型
mise run translate -- "book.epub" --model "hf.co/tencent/Hy-MT2-7B-GGUF:Q4_K_M"

# 分别指定翻译模型和摘要模型
mise run translate -- "book.epub" --model "translategemma:12b" --summary-model "qwen3.5:9b"

# 已安装全局命令时同样支持这些选项
translate-books "book.epub" --model "你的模型名称" --summary-model "你的摘要模型名称"
```

模型名称必须与 `ollama list` 一致；模型缺失时会报错，不会自动下载或替换模型。目前提供 Ollama 接口，不支持直接传入 GGUF 文件或任意 OpenAI 兼容接口。

## 其他用法

```bash
# 仅检查文本和章节，不加载模型、不写文件
mise run translate -- "book.epub" --dry-run

# 先看摘要；后续执行完整翻译时自动复用缓存
mise run translate -- "book.epub" --summary-only

# 可选：使用另一个本地模型概括，翻译仍使用指定 Hy-MT2 模型
mise run translate -- "book.epub" --summary-model qwen3.5:9b

# 更换目标语言
mise run translate -- "book.epub" -t zh-Hant
mise run translate -- "book.epub" -t en
mise run translate -- "book.epub" -t 葡萄牙语 --lang-code pt

# 自定义 Ollama 服务与上下文窗口
mise run translate -- "book.epub" --host http://localhost:11434 --num-ctx 16384
```

## 输出与恢复

默认在原书旁生成：

- `book.zh-Hans.epub`：译本，正文、章节标题、HTML 目录和 NCX 阅读器目录译为目标语言。
- `book.zh-Hans.summary.md`：全书与各章摘要、模型建议的术语译法。
- `book.zh-Hans.translate-cache.sqlite3`：逐请求缓存，用于中断恢复。

按 Ctrl+C 后重新执行同一命令即可复用已完成请求。请求键包含完整提示词、模型摘要标识和生成参数；修改输入、摘要或参数后，对应请求会重新计算。网络错误和服务端临时错误默认重试两次；模型返回空结果、输出截断或翻译失败会报错，不会生成看似完成的部分译本。摘要文件会在翻译开始前保存；重跑会更新同名摘要。

已有译本需要 `--force` 才能覆盖，原书始终不能作为输出。完整翻译成功并校验 ZIP 后才原子发布译本。图片、CSS、字体、链接和章节顺序保留；行内 XML 格式经过结构检查，只写回文本。结构检查失败时使用原文段落作为额外上下文，回退到逐文本节点翻译。超长段落按句子或空白分块，极长无分隔文本按字符上限切分。

## 参数与限制

默认上下文为 16384 tokens，翻译分块 1200 字符，摘要输入块 2500 字符，全书摘要最多 1000 字符。内部按 UTF-8 字节数保守预留输入和输出空间，分块可能小于配置值；超出窗口时会提示调整，避免静默丢失上下文。可用 `--chunk-chars`、`--summary-chunk-chars`、`--summary-chars` 调整速度和上下文大小。长书先概括再翻译需要较多模型请求，CLI 会报告章节与段落进度。

进度日志显示本机当前时间、时区、本次运行已用时间。正文翻译开始后，还会按段落刷新完成数量、百分比、预计剩余时间和预计完成时刻：

```text
[当前 2026-09-05 10:48:00 +0800] 翻译 14/29 · ... · 段落 32/76 | 进度 358/1532 (23.4%) | 本次已用 00:10:00 | 预计剩余 00:15:20 · 预计完成 09-05 11:03:20
```

预计剩余时间根据最近 50 个实际调用模型的段落耗时、文本长度和并发数计算，包含尚未翻译的正文和阅读器目录。前 3 个新段落完成前显示“估算中”；缓存命中会增加完成进度，但不会用于估算模型速度。重启后“本次已用”重新计时。摘要阶段只显示当前时间和已用时间；`--quiet` 隐藏这些进度日志。

正文及阅读器目录默认以 2 并发翻译，使用 `--workers 1` 可改为串行，摘要仍按依赖顺序生成。程序只保留最多指定数量的在途任务；各任务翻译独立副本，再按原位置写回，完成先后不会改变章节和段落顺序。SQLite 缓存支持线程安全写入，切换并发数不会使已有缓存失效。

Ollama 服务端也需要允许相应的并行请求，例如启动服务时设置 `OLLAMA_NUM_PARALLEL=2`。仅给已经启动的客户端设置该环境变量无效；具体设置及显存要求见 [Ollama 并发说明](https://docs.ollama.com/faq#how-does-ollama-handle-concurrent-requests)。进度中的并发数表示 CLI 任务数，实际模型吞吐量取决于服务端配置与可用显存。

适用于可提取文本、XHTML 合法的 EPUB 2/3。图片里的文字不会 OCR 或翻译；代码、公式、SVG、`translate="no"` 内容保留。图像替代文本及其他属性、OPF 书名和作者等元数据保留，语言字段更新。内嵌字体若缺少中文字形，阅读器需要提供中文回退字体；固定版式在译文变长后仍可能需要人工检查。暂不支持正文 DRM、数字签名、多 rendition 或不合法 XHTML。

摘要和术语由模型生成，复杂专名仍建议人工核对。默认摘要和翻译都使用你指定的模型；`--summary-model` 可以选用已安装的其他本地模型。全部书籍内容发送到所配置的 Ollama 服务，默认仅 `localhost`，没有云端翻译接口。

提示词结合了 [Hy-MT2 官方的上下文与结构化翻译方式](https://huggingface.co/tencent/Hy-MT2-7B-GGUF)，调用 [Ollama Generate API](https://docs.ollama.com/api/generate)。

## 开发

```bash
mise run test
mise run lint
```
