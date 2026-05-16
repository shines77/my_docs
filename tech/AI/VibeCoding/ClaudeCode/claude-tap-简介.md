# claude-tap 简介

## 简介

claude-tap 是一个用于拦截和调试 Claude AI 工具（如 Claude Code、Codex CLI、OpenCode、Cursor CLI）API 请求的开发工具。它通过代理模式捕获客户端与 AI 服务之间的通信流量，并在浏览器中以可视化 Trace Viewer 展示请求结构、系统提示、工具调用、token 使用等关键信息，便于调试和优化 AI 交互逻辑。

---

## 核心功能

- 拦截 API 流量：捕获 Claude Code、Codex CLI、Cursor CLI 等工具发出的请求。
- 结构化查看请求内容：
  - 系统提示（system prompts）
  - 对话历史
  - 工具调用（tool selections）
  - token 消耗详情
- 支持多种视图模式：
  - Live Viewer：实时浏览器视图，随请求动态更新。
  - Diff View：对比连续请求的变化（字符级高亮）。
  - Path Filtering：按 API 路径（如 `/v1/messages`）过滤请求。
  - Model Grouping：按模型（如 claude-sonnet、claude-opus）分组显示。
- 支持反向代理与正向代理模式：适用于不同集成场景（如对接 DeepSeek、自定义后端）。

---

## 安装与使用

安装方式（推荐）

```bash
使用 uv（推荐）
uv tool install claude-tap

或使用 pip
pip install claude-tap
```

> 要求：Python 3.11+，以及要追踪的 CLI（如 Claude Code、Cursor 等）已安装 。

快速启动示例

```bash
捕获 Claude Code 默认请求
claude-tap

启用实时浏览器视图
claude-tap --tap-live

捕获 Cursor CLI 请求
claude-tap --tap-client cursor -- -p --trust --model auto "hello"

跳过权限提示（仅限测试）
claude-tap -- --dangerously-skip-permissions
```

---

## 高级用法

- 对接 DeepSeek API：
  ```bash
  export ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic
  claude-tap --tap-proxy-mode reverse --tap-target https://api.deepseek.com/anthropic
  ```

- 独立查看历史 Trace：
  ```bash
  claude-tap export .traces/2026-02-28/trace_141557.jsonl -o trace.html
  ```

- Dashboard 模式（不启动客户端）：
  ```bash
  claude-tap dashboard --tap-live-port 3000
  ```

---

## 适用场景

- 调试 AI 工具为何生成冗长输出或错误工具调用。
- 优化系统提示以减少 token 消耗。
- 学习 Claude 系列模型如何构建请求与管理上下文。
- 集成自定义后端（如 DeepSeek、本地代理）时验证请求格式。

## GitHub

> 项目地址：[GitHub - liaohch3/claude-tap](https://github.com/liaohch3/claude-tap)
