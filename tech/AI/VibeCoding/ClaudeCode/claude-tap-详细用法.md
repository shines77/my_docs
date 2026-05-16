# claude-tap 详细用法

## claude-tap 是什么

`calude-tap` 是一个 Coding Agent CLI 的 API 流量拦截器。

它可以拦截并查看 `Claude Code`、`Codex CLI`、`Gemini CLI`、`Kimi CLI`、`OpenCode`、`Pi`、`Hermes Agent` 或 `Cursor CLI` 的所有 API 流量。看清它们如何构造 `system prompt`、`管理对话历史`、`选择工具`、`优化 token 用量` —— 通过一个美观的 `trace` 查看器。

Github：[https://github.com/liaohch3/claude-tap](https://github.com/liaohch3/claude-tap)

## 工作原理

- `claude-tap` 启动反向代理或 `forward proxy`，并启动所选客户端
- 支持 `base URL` 的客户端会指向反向代理；不支持 `base URL` 的客户端会通过 `proxy/CA` 环境变量接入
- `SSE` 和 `WebSocket` 流会在收到 `chunk/message` 时实时转发，代理开销很低
- 每个请求-响应对或 `WebSocket` 会话记录到按日期保存的 `trace_*.jsonl`
- 退出时生成自包含的 `HTML` 查看器
- 实时模式（可选）通过 `SSE` 向浏览器广播更新

核心特性: 🔒 常见认证 header 自动脱敏 · ⚡ 低开销流式转发 · 📦 自包含查看器 · 🔄 实时模式

## 如何安装 cluade-tap

要求：`Python 3.11+` 环境，使用 uv 或 pip 安装。

```bash
# 推荐
uv tool install claude-tap

# 或用 pip
pip install claude-tap
````

## 如何升级 cluade-tap

使用如下命令：

```bash
# 推荐
claude-tap update

# 使用 uv
uv tool upgrade claude-tap

# 或用 pip
pip install --upgrade claude-tap
````

## 快速开始

可拦截的 CLI 以及相应的命令：

- `Claude Code`（默认）
- `Codex CLI`（--tap-client codex）
- `Gemini CLI`（--tap-client gemini）
- `Kimi CLI`（--tap-client kimi）
- `OpenCode`（--tap-client opencode）
- `Pi`（--tap-client pi）
- `Hermes Agent`（--tap-client hermes）
- `Cursor CLI`（--tap-client cursor）

用 claude-tap 启动你想观察的客户端：

```bash
# Claude Code
claude-tap

# Claude Code + 浏览器实时查看
claude-tap --tap-live

# Codex CLI
claude-tap --tap-client codex

# Gemini CLI
claude-tap --tap-client gemini -- -p "hello"

# Kimi CLI
claude-tap --tap-client kimi

# Pi
claude-tap --tap-client pi -- --model openai-codex/gpt-5.3-codex-spark -p "hello"

# Cursor CLI
claude-tap --tap-client cursor -- -p --trust --model auto "hello"
```

注：非 `--tap-*` 的参数且跟在 `--` 之后的参数，会传递给所选客户端。

## Claude Code 详细用法

```bash
# 透传参数给 Claude Code
claude-tap -- --model claude-opus-4-6
# 继续上次对话
claude-tap -c
# 或
claude-tap -- -c

# 跳过所有权限确认（自动批准工具调用）
claude-tap -- --dangerously-skip-permissions

# 实时查看器 + 跳过权限确认 + 指定模型
claude-tap --tap-live -- --dangerously-skip-permissions --model claude-sonnet-4-6
```

## Codex CLI 详细用法

Codex CLI 支持两种认证方式：OAuth 和 API Key，常用的是 OAuth 方式。

| 认证方式 | 如何认证 | 上游目标 |   说明   |
| :------: | :------: | :------: | :------: |
| OAuth（ChatGPT 付费套餐）| codex login | https://chatgpt.com/backend-api/codex | ChatGPT Plus/Pro/Team 用户默认方式 |
| API Key | 设置 OPENAI_API_KEY | https://api.openai.com（默认） | 通过 OpenAI Platform 按量付费 |

### OAuth 认证

即 ChatGPT 付费套餐，例如：Free，Plus，Pro，Team 等。使用 `codex login` 命令后通常会自动识别。

```bash
# OAuth 用户（ChatGPT Plus/Pro/Team）
claude-tap --tap-client codex

# 如果无法读取 Codex auth 文件，可以显式指定 target
claude-tap --tap-client codex --tap-target https://chatgpt.com/backend-api/codex

# 指定模型
claude-tap --tap-client codex -- --model codex-mini-latest

# 全自动模式（跳过所有权限确认）
claude-tap --tap-client codex -- --full-auto

# OAuth + 全自动 + 实时查看器
claude-tap --tap-client codex --tap-live -- --full-auto
```

### API Key 认证

需要设置 `OPENAI_API_KEY`，Base URL 是 `https://api.openai.com`（默认的，设置 `OPENAI_API_KEY` 后生效）。

claude-tap 会尽量根据 Codex 的认证状态自动识别 target。

```bash
# API Key 用户 — 默认 OpenAI API target 即可
claude-tap --tap-client codex
```

## OpenCode 示例

[OpenCode](https://opencode.ai/) 是一款多 provider 的终端 AI 助手。由于它能对接多种 provider，claude-tap 默认对 opencode 使用 forward proxy 模式——向子进程注入 `HTTPS_PROXY` 与本地 CA，捕获它对接的任意 provider 流量。

```bash
# forward proxy 模式 — 捕获 opencode 对接的任意 provider（默认）
claude-tap --tap-client opencode

# 实时查看器
claude-tap --tap-client opencode --tap-live

# reverse 模式 — 仅在使用 Anthropic provider 时有效（单一 ANTHROPIC_BASE_URL）
claude-tap --tap-client opencode --tap-proxy-mode reverse
```

## Kimi CLI 示例

Kimi CLI 默认通过 `KIMI_BASE_URL` 使用 reverse proxy。使用你已有的 Kimi CLI 认证和配置；默认上游目标是 Kimi Code API。

```bash
# reverse proxy 模式
claude-tap --tap-client kimi
claude-tap --tap-client kimi -- --thinking

# 实时查看器
claude-tap --tap-client kimi --tap-live -- --thinking

# 改用 Moonshot Open Platform，而不是 Kimi Code
claude-tap --tap-client kimi --tap-target https://api.moonshot.ai/v1
```

## Gemini CLI 示例

Gemini CLI 默认使用 `forward proxy`。Google OAuth / Code Assist 流量会访问多个 Google 端点，因此 forward proxy 是更稳妥的默认抓取方式。对于会读取 `GOOGLE_GEMINI_BASE_URL` 或 `GOOGLE_VERTEX_BASE_URL` 的 API key / Vertex 类流程，仍可显式使用 reverse 模式。

```bash
# Google OAuth / Code Assist
claude-tap --tap-client gemini -- -p "hello"

# 实时查看器
claude-tap --tap-client gemini --tap-live -- -p "hello"

# API key / Vertex 兼容流程的 reverse 模式
claude-tap --tap-client gemini --tap-proxy-mode reverse -- -p "hello"
```

## 浏览器预览、导出

```bash
# 禁用退出后自动打开 HTML 查看器（默认开启）
claude-tap --tap-no-open

# 实时模式 — 客户端运行时在浏览器中实时查看
claude-tap --tap-live
claude-tap --tap-live --tap-live-port 3000    # 固定实时查看器端口

# 独立 Dashboard — 不启动客户端，直接浏览历史 trace
claude-tap dashboard
claude-tap dashboard --tap-output-dir ./my-traces --tap-live-port 3000
```

客户端退出后，也可以手动打开生成的查看器：

```bash
open .traces/*/trace_*.html
```

也可以从已有 JSONL trace 重新生成自包含 HTML 查看器：

```bash
claude-tap export .traces/2026-02-28/trace_141557.jsonl -o trace.html
# 或：
claude-tap export .traces/2026-02-28/trace_141557.jsonl --format html
```

## 纯代理模式

仅启动代理，不自动启动客户端 — 适用于自定义场景或在另一个终端手动连接：

```bash
# Claude Code
claude-tap --tap-no-launch --tap-port 8080
# 在另一个终端:
ANTHROPIC_BASE_URL=http://127.0.0.1:8080 claude

# Anthropic Python SDK（或任何基于它构建的自定义 Agent）
claude-tap --tap-no-launch --tap-port 8080
# 在你的 Agent 进程中:
ANTHROPIC_BASE_URL=http://127.0.0.1:8080 python your_agent.py

# Codex CLI（OAuth）
claude-tap --tap-client codex --tap-target https://chatgpt.com/backend-api/codex --tap-no-launch --tap-port 8080
# 在另一个终端:
OPENAI_BASE_URL=http://127.0.0.1:8080/v1 codex -c 'openai_base_url="http://127.0.0.1:8080/v1"'

# Codex CLI（API Key）
claude-tap --tap-client codex --tap-no-launch --tap-port 8080
# 在另一个终端:
OPENAI_BASE_URL=http://127.0.0.1:8080/v1 codex -c 'openai_base_url="http://127.0.0.1:8080/v1"'

# Kimi CLI
claude-tap --tap-client kimi --tap-no-launch --tap-port 8080
# 在另一个终端:
KIMI_BASE_URL=http://127.0.0.1:8080 kimi

# Gemini CLI（仅 reverse 模式）
claude-tap --tap-client gemini --tap-proxy-mode reverse --tap-no-launch --tap-port 8080
# 在另一个终端:
GOOGLE_GEMINI_BASE_URL=http://127.0.0.1:8080 GOOGLE_VERTEX_BASE_URL=http://127.0.0.1:8080 gemini
```

## 常用组合

```bash
# 追踪 Claude Code：实时查看器 + 自动批准
claude-tap --tap-live -- --dangerously-skip-permissions

# 追踪 Codex（OAuth）：实时查看器 + 全自动
claude-tap --tap-client codex --tap-target https://chatgpt.com/backend-api/codex --tap-live -- --full-auto

# 自定义 trace 输出目录
claude-tap --tap-output-dir ./my-traces

# 仅保留最近 10 次 trace
claude-tap --tap-max-traces 10
```

## CLI 选项

除以下 `--tap-*` 参数外，所有 `--` 之后的参数均传递给所选客户端：

```bash
--tap-client CLIENT      启动的客户端: claude（默认）/ codex / gemini / kimi / opencode / pi / hermes / cursor
--tap-target URL         上游 API 地址（默认: 根据客户端自动选择）
--tap-live               启动实时查看器（自动打开浏览器）
--tap-live-port PORT     实时查看器端口（默认: 自动分配）
--tap-no-open            退出后不自动打开 HTML 查看器（默认开启）
--tap-output-dir DIR     Trace 输出目录（默认: ./.traces）
--tap-port PORT          代理端口（默认: 自动分配）
--tap-host HOST          绑定地址（默认: 127.0.0.1，--tap-no-launch 模式下为 0.0.0.0）
--tap-no-launch          仅启动代理，不启动客户端
--tap-max-traces N       最大保留 trace 数量（默认: 50，0 = 不限）
--tap-no-update-check    禁用启动时的 PyPI 更新检查
--tap-no-auto-update     仅检查更新，不自动下载
--tap-proxy-mode MODE    代理模式: reverse 或 forward（默认：claude/codex/kimi 用 reverse，gemini/opencode/pi/hermes/cursor 用 forward）
```

## 查看器功能

查看器是一个自包含的 HTML 文件（零外部依赖）：

- **结构化 Diff** — 对比相邻请求的变化：新增/删除的消息、system prompt diff、字符级高亮
- **路径过滤** — 按 API 端点筛选（如仅显示 /v1/messages）
- **模型分组** — 侧边栏按模型分组，并对 Claude 系列模型做优先排序
- **Token 用量分析** — 输入 / 输出 / 缓存读取 / 缓存创建
- **工具检查器** — 可展开的卡片，显示工具名称、描述和参数 schema
- **全文搜索** — 搜索消息、工具、prompt 和响应
- **暗色模式** — 切换亮色/暗色主题（跟随系统偏好）
- **键盘导航** — j/k 或方向键
- **复制助手** — 一键复制请求 JSON 或 cURL 命令
- **多语言** — English, 简体中文, 日本語, 한국어, Français, العربية, Deutsch, Русский

## 相关链接

- [https://github.com/liaohch3/claude-tap](https://github.com/liaohch3/claude-tap)

## 参考文献

- [https://github.com/liaohch3/claude-tap/blob/main/README_zh.md](https://github.com/liaohch3/claude-tap/blob/main/README_zh.md)
