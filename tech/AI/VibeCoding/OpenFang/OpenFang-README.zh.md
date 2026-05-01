# OpenFang

## The Agent Operating System

Open-source Agent OS built in Rust. 137K LOC. 14 crates. 1,767+ tests. Zero clippy warnings.
**One binary. Battle-tested. Agents that actually work for you.**

[Documentation](https://openfang.sh/docs) • [Quick Start](https://openfang.sh/docs/getting-started) • [Twitter / X](https://x.com/openfangg)

![Rust](https://img.shields.io/badge/language-Rust-orange?style=flat-square) ![MIT](https://img.shields.io/badge/license-MIT-blue?style=flat-square) ![v0.6.2](https://img.shields.io/badge/version-0.6.2-green?style=flat-square) ![Tests](https://img.shields.io/badge/tests-2,517%2B%20passing-brightgreen?style=flat-square) ![Clippy](https://img.shields.io/badge/clippy-0%20warnings-brightgreen?style=flat-square) [![Buy Me A Coffee](https://img.shields.io/badge/Buy%20Me%20a%20Coffee-FFDD00?style=flat-square&logo=buy-me-a-coffee&logoColor=black)](https://www.buymeacoffee.com/openfang)

---

> **v0.5.10（2026 年 4 月）**
>
> OpenFang 功能完整，但仍然是 1.0 之前的版本。预计次要版本之间会有粗糙的边缘和突破性的变化。我们发货快，修理快。固定到特定的提交以供生产使用，直到 v1.0。 [在此处报告问题。](https://github.com/RightNow-AI/openfang/issues)

---

## 什么是 OpenFang ？

OpenFang 是一个 **开源 Agent 操作系统**。不是聊天机器人框架。不是围绕 LLM 的 Python 包装器。不是一个“multi-agent 编排器”。一个完整的自主 Agent 操作系统，用 Rust 从头开始构建。

传统的 agent 框架等待您键入内容。OpenFang 运行 **为您工作的自主 agent**：按计划，全天候，构建知识图，监控目标，生成潜在客户，管理社交媒体，并将结果报告到仪表板。

整个系统编译为 **单个~32MB二进制文件**。一次安装，一个命令，你的 Agents 就上线了。

BASH

```bash
curl -fsSL https://openfang.sh/install | sh
openfang init
openfang start
# Dashboard live at http://localhost:4200
```

**Windows**

POWERSHELL

```powershell
irm https://openfang.sh/install.ps1 | iex
openfang init
openfang start
```

---

## Hands：真正做事的 Agents

_"Traditional agents wait for you to type. Hands work **for** you."_

**Hands** 是 OpenFang 的核心创新。预构建的自主功能包，可按计划独立运行，无需提示。这不是聊天机器人。这是一个代理，它在早上 6 点醒来，研究你的竞争对手，构建知识图，对结果进行评分，并在你喝咖啡之前向你的 Telegram 发送一份报告。

每只 Hand bundles：

- **HAND.toml**：清单声明工具、设置、需求和仪表板指标。

- **系统提示**：多阶段行动手册，不是一行文字。这些是 500 多字的专家程序。

- **SKILL.md**：在运行时将领域专业知识参考注入上下文。

- **护栏**：敏感操作的批准门（例如，Browser Hand 在任何购买前都需要批准）。

全部编译成二进制文件。无需下载，无需 pip 安装，无需 Docker 拉取。

### 7 个 Bundled Hands

| .Hand. | 它到底在做什么 |
| --- | --- |
| **Clip** | 剪辑，获取 YouTube URL，下载它，识别最佳时刻，将其切成带有字幕和缩略图的垂直短片，可选择添加 AI 画外音，并发布到 Telegram 和 WhatsApp。8 相管道。FFmpeg + yt-dlp + 5 STT 后端。 |
| **Lead** | 领导者，每天跑步。发现与您的 ICP 匹配的潜在客户，用网络研究丰富他们，评分 0-100，根据您现有的数据库进行重复数据消除，并以 CSV/JSON/Markdown 格式提供合格的潜在客户。随着时间的推移建立 ICP 档案。 |
| **Collector** | 收集者，OSINT 级智力。你给它一个目标（公司、个人、主题）。它持续监控：变化检测、情绪跟踪、知识图构建，以及重要事物发生变化时的关键警报。 |
| **Predictor** | 预测器，超级预测引擎。从多个来源收集信号，构建校准的推理链，使用置信区间进行预测，并使用 Brier 评分跟踪其自身的准确性。有一种逆向思维模式，故意反对共识。 |
| **Researcher** | 研究员，深度自主研究者。交叉引用多个来源，使用 CRAAP 标准（货币、相关性、权威性、准确性、目的）评估可信度，生成 APA 格式的引用报告，支持多种语言。 |
| **Twitter** | 推特，自主推特/X 账户管理器。以 7 种轮换格式创建内容，安排帖子以获得最佳参与度，回应提及，跟踪绩效指标。有一个审批队列，所以没有你的同意，什么都不会发布 |
| **Browser** | 浏览器，Web 自动化代理。浏览网站、填写表单、点击按钮、处理多步骤工作流。使用具有会话持久性的 Playwright 桥。 **强制性采购审批门**：没有明确的确认，它永远不会花你的钱。 |

BASH

```bash
# Activate the Researcher Hand. It starts working immediately.
openfang hand activate researcher
# Check its progress anytime
openfang hand status researcher
# Activate lead generation on a daily schedule
openfang hand activate lead
# Pause without losing state
openfang hand pause lead
# See all available Hands
openfang hand list
```

**建立自己的。** 定义一个 `HAND.toml` 使用工具、设置和系统提示。发布到 FangHub。

---

## OpenFang vs 景观

### 基准测试：已测量，不是营销

所有数据来自官方文件和公共存储库，2026 年 2 月。

#### 冷启动时间（越短越好）

```bash
ZeroClaw   ██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   10 ms
OpenFang   ██████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  180 ms    ★
LangGraph  █████████████████░░░░░░░░░░░░░░░░░░░░░░░░░  2.5 sec
CrewAI     ████████████████████░░░░░░░░░░░░░░░░░░░░░░  3.0 sec
AutoGen    ██████████████████████████░░░░░░░░░░░░░░░░░  4.0 sec
OpenClaw   █████████████████████████████████████████░░  5.98 sec
```

#### 空闲内存使用率（越低越好）

```bash
ZeroClaw   █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░    5 MB
OpenFang   ████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   40 MB    ★
LangGraph  ██████████████████░░░░░░░░░░░░░░░░░░░░░░░░░  180 MB
CrewAI     ████████████████████░░░░░░░░░░░░░░░░░░░░░░░  200 MB
AutoGen    █████████████████████████░░░░░░░░░░░░░░░░░░  250 MB
OpenClaw   ████████████████████████████████████████░░░░  394 MB
```

#### 安装尺寸（越小越好）

```bash
ZeroClaw   █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  8.8 MB
OpenFang   ███░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   32 MB    ★
CrewAI     ████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  100 MB
LangGraph  ████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  150 MB
AutoGen    ████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░  200 MB
OpenClaw   ████████████████████████████████████████░░░░  500 MB
```

#### 安全系统（越高越好）

```bash
OpenFang   ████████████████████████████████████████████   16      ★
ZeroClaw   ███████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░    6
OpenClaw   ████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░    3
AutoGen    █████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░    2
LangGraph  █████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░    2
CrewAI     ███░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░    1
```

#### 通道适配器（越高越好）

```bash
OpenFang   ████████████████████████████████████████████   40      ★
ZeroClaw   ███████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░   15
OpenClaw   █████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   13
CrewAI     ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░    0
AutoGen    ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░    0
LangGraph  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░    0
```

#### LLM 提供者（越高越好）

```bash
ZeroClaw   ████████████████████████████████████████████   28
OpenFang   ██████████████████████████████████████████░░   27      ★
LangGraph  ██████████████████████░░░░░░░░░░░░░░░░░░░░░   15
CrewAI     ██████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   10
OpenClaw   ██████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   10
AutoGen    ███████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░    8
```

### 逐个特征比较

|功能|OpenFang|OpenClaw|ZeroClaw|CrewAI|AutoGen|LangGraph|
|---|---|---|---|---|---|---|
|**语言**|**Rust**|TypeScript|**Rust**|Python|Python|Python|
|**Autonomous Hands**|**7个内置**|无|无|无|||
|**安全层**|**16个离散**|3个基本|6层|1个基本|Docker|AES加密|
|**Agent Sandbox**|**WASM dual-metered**|无|允许列表|无|Docker|无|
|**Channel 适配器**|**40**|13|15|0|0|0|
|**内置工具**|**53+MCP+A2A**|50+|12|插件|MCP|LC工具|
|**记忆**|**SQLite+矢量**|基于文件|SQLite FTS5|4层|外部|检查点|
|**桌面应用程序**|**Tauri 2.0**|无|无|无|Studio|无|
|**审计跟踪**|**Merkle hash-chain**|日志|日志|跟踪|日志|检查点|
|**冷启动**|**<200ms**|~6s|~10ms|~3s|~4s|~2.5s|
|**安装尺寸**|**约32 MB**|约500 MB|约8.8 MB|约100 MB|约200 MB|约150 MB|
|**开源协议**|MIT|MIT|MIT|MIT|Apache 2.0|MIT|

---

## 16 个安全系统：深度防御

OpenFang 不会在事后加强安全措施。每一层都是可独立测试的，并且在没有单点故障的情况下运行。

|#|系统|它做什么|
|---|---|---|
|1|**WASM dual-metered 沙箱**|工具代码在 WebAssembly 中运行，带有 fuel metering + epoch interruption 。watchdog 线程会关掉失控的代码。|
|2|**Merkle hash-chain 审计追踪**|每一个动作都与前一个动作有加密联系。篡改一个入口，整个链条就会断裂。|
|3|**信息流跟踪**|标签通过执行传播。秘密从源头追踪到源头。|
|4|**Ed25519 签名代理人清单**|每个代理身份和能力集都经过加密签名。|
|5|**SSRF Protection**|阻止私有 IP、云元数据端点和 DNS 重新绑定攻击。|
|6|**Secret Zeroization**|`Zeroizing<String>` 在不再需要 API 密钥时，自动从内存中清除这些密钥。|
|7|**OFP 相互认证**|用于P2P网络的基于 HMAC-SHA256 随机数的恒定时间验证。|
|8|**能力门**|基于角色的访问控制。代理声明所需的工具，内核强制执行|
|9|**安全标头**|每个响应上都有 CSP、X-Frame-Options、HSTS、X-Content-Type-Options。|
|10|**健康端点补救措施**|公共卫生检查返回的信息很少。完整的诊断需要身份验证。|
|11|**子流程沙盒**|`env_clear()` + 选择性变量传递。使用跨平台 kill 进行进程树隔离。|
|12|**快速注射扫描仪**|检测技能中的覆盖尝试、数据泄露模式和 shell 引用注入。|
|13|**环路保护**|基于 SHA256 的断路器工具调用回路检测。处理乒乓球图案。|
|14|**会话修复**|7 阶段消息历史验证和从损坏中自动恢复。|
|15|**路径穿越预防**|规范化，防止符号链接逃逸。 `../` 这里不工作。|
|16|**GCRA 速率限制器**|具有成本意识的令牌桶速率限制，具有每 IP 跟踪和过时清理功能。|

---

## Architecture（架构）

14 个 Rust crates，137728 行代码。模块化内核设计。

```bash
openfang-kernel      Orchestration, workflows, metering, RBAC, scheduler, budget tracking
openfang-runtime     Agent loop, 3 LLM drivers, 53 tools, WASM sandbox, MCP, A2A
openfang-api         140+ REST/WS/SSE endpoints, OpenAI-compatible API, dashboard
openfang-channels    40 messaging adapters with rate limiting, DM/group policies
openfang-memory      SQLite persistence, vector embeddings, canonical sessions, compaction
openfang-types       Core types, taint tracking, Ed25519 manifest signing, model catalog
openfang-skills      60 bundled skills, SKILL.md parser, FangHub marketplace
openfang-hands       7 autonomous Hands, HAND.toml parser, lifecycle management
openfang-extensions  25 MCP templates, AES-256-GCM credential vault, OAuth2 PKCE
openfang-wire        OFP P2P protocol with HMAC-SHA256 mutual authentication
openfang-cli         CLI with daemon management, TUI dashboard, MCP server mode
openfang-desktop     Tauri 2.0 native app (system tray, notifications, global shortcuts)
openfang-migrate     OpenClaw, LangChain, AutoGPT migration engine
xtask                Build automation
```

---

## 40 个 Channel 适配器

将您的代理连接到您的用户所在的每个平台。

- **Core（核心）：** Telegram、Discord、Slack、WhatsApp、Signal、Matrix、Email（IMAP/SMTP）
- **Enterprise（企业）：** Microsoft Teams、Mattermost、Google Chat、Webex、Feishu/Lark、Zulip
- **Social（社交媒体）：** LINE、Viber、Facebook Messenger、Mastodon、Bluesky、Reddit、LinkedIn、Twitch
- **Community（社区）：** IRC、XMPP、Guilded、Revolt、Keybase、Discourse、Gitter
- **Privacy（隐私）：** Threema、Nost、Mumble、Nextcloud Talk、Rocket.Chat、Ntfy、Gotify
- **Workplace（工作空间）：** Pumble、Flock、Twist、DingTalk、Zalo、Webhooks

每个适配器都支持每个 Channel 模型覆盖、DM/group 策略、速率限制和输出格式。

---

## WhatsApp 网络网关（二维码）

通过二维码将您的个人 WhatsApp 帐户连接到 OpenFang，就像 WhatsApp Web 一样，不需要 Meta Business 帐户。

### 先决条件

- **Node.js >= 18** 已安装([下载](https://nodejs.org/))

- OpenFang 已安装并初始化

### 设置

**1. 安装网关依赖项：**

BASH

```bash
cd packages/whatsapp-gateway
npm install
```

**2. 配置 `config.toml`:**

TOML

```toml
[channels.whatsapp]
mode = "web"
default_agent = "assistant"
```

**3. 设置网关URL（选择一个）：**

添加到您的 shell 配置文件中以实现持久性：

BASH

```bash
# macOS / Linux
echo 'export WHATSAPP_WEB_GATEWAY_URL="http://127.0.0.1:3009"' >> ~/.zshrc
source ~/.zshrc
```

或者在启动网关时将其设置为内联：

BASH

```bash
export WHATSAPP_WEB_GATEWAY_URL="http://127.0.0.1:3009"
```

**4. 启动网关：**

BASH

```bash
node packages/whatsapp-gateway/index.js
```

网关监听端口 `3009` 默认情况下。覆盖 `WHATSAPP_GATEWAY_PORT`.

**5. 启动OpenFang：**

BASH

```bash
openfang start
# Dashboard at http://localhost:4200
```

**6. 扫描二维码：**

打开仪表板→ **Channel** → **WhatsApp**。将出现一个二维码。用手机扫描：

> **WhatsApp** → **设置** → **链接设备** → **链接设备**

扫描后，状态将更改为 `connected` 传入的消息会被路由到您配置的代理。

### 网关环境变量

|变量|描述|默认值|
|---|---|---|
|`WHATSAPP_WEB_GATEWAY_URL`|OpenFang要连接的网关URL|_（空=禁用）_|
|`WHATSAPP_GATEWAY_PORT`|网关监听的端口|`3009`|
|`OPENFANG_URL`|网关向其报告的OpenFang API URL|`http://127.0.0.1:4200`|
|`OPENFANG_DEFAULT_AGENT`|处理传入消息的代理|`assistant`|

### 网关 API 端点

|方法|路线|描述|
|---|---|---|
|`POST`|`/login/start`|生成二维码（返回base64 PNG）|
|`GET`|`/login/status`|连接状态(`disconnected`, `qr_ready`, `connected`)|
|`POST`|`/message/send`|发送消息(`{ "to": "5511999999999", "text": "Hello" }`)|
|`GET`|`/health`|健康检查|

### 备选方案：WhatsApp 云 API

对于生产工作负载，请使用 [WhatsApp云API](https://developers.facebook.com/docs/whatsapp/cloud-api) 拥有Meta Business帐户。看 [云API配置文档](https://openfang.sh/docs/channels/whatsapp).

---

## 27 个 LLM 提供商，123+ 个模型

通过 3 种不同协议（Anthropic、Gemini、OpenAI 兼容）路由到 27 家供应商：

Anthropic、Gemini、OpenAI、Groq、DeepSeek、OpenRouter、Together、Mistral、Fireworks、Cohere、Perplexity、xAI、AI21、Cerebras、SambaNova、HuggingFace、Replicate、Ollama、vLLM、LM Studio、Qwen、MiniMax、Zhipu、Moonshot、Qianfan、Bedrock 等。

具有任务复杂性评分、自动回退、成本跟踪和按型号定价的智能路由。

---

## 从 OpenClaw 迁移

已经在运行 OpenClaw 了？输入一个命令：

BASH

```bash
# Migrate everything: agents, memory, skills, configs.
openfang migrate --from openclaw
# Migrate from a specific path
openfang migrate --from openclaw --path ~/.openclaw
# Dry run first to see what would change
openfang migrate --from openclaw --dry-run
```

迁移引擎导入您的代理、对话历史、技能和配置。[OpenFang 原生读取 SKILL.md](http://xn--openfangskill-3g7v7ym542bohsd.md/)，并与 ClawHub 市场兼容。

---

## OpenAI兼容的API

插入替换。将您现有的工具指向 OpenFang：

BASH

```bash
curl -X POST localhost:4200/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "researcher",
    "messages": [{"role": "user", "content": "Analyze Q4 market trends"}],
    "stream": true
  }'
```

140 多个 REST/WS/SSE 端点，涵盖 Agents、Memory、Workflows、Channel、Models、Skills、A2A、Hands 等。

---

## 快速开始

BASH

```bash
# 1. Install (macOS/Linux)
curl -fsSL https://openfang.sh/install | sh
# 2. Initialize. Walks you through provider setup.
openfang init
# 3. Start the daemon
openfang start
# 4. Dashboard is live at http://localhost:4200
# 5. Activate a Hand. It starts working for you.
openfang hand activate researcher
# 6. Chat with an agent
openfang chat researcher
> "What are the emerging trends in AI agent frameworks?"
# 7. Spawn a pre-built agent
openfang agent spawn coder
```

**Windows (PowerShell)**

POWERSHELL

```powershell
irm https://openfang.sh/install.ps1 | iex
openfang init
openfang start
```

---

## 发展

BASH

```bash
# Build the workspace
cargo build --workspace --lib
# Run all tests (1,767+)
cargo test --workspace
# Lint (must be 0 warnings)
cargo clippy --workspace --all-targets -- -D warnings
# Format
cargo fmt --all -- --check
```

---

## 稳定性通知

OpenFang v0.5.10 是 1.0 之前的版本。架构坚实，测试套件全面，安全模型深入。上面说：

- **突破性变化** 可能发生在 v1.0 之前的次要版本之间。

- **一些 Hand** 他们比其他人更成熟。浏览器和研究人员是最久经考验的。

- **边缘案例** 存在。如果你找到一个， [打开一个问题](https://github.com/RightNow-AI/openfang/issues).

- **固定到特定的提交** 适用于 v1.0 之前的生产部署。

我们发货快，修理快。目标是在 2026 年中期推出坚如磐石的 v1.0。

---

## 安全

要报告安全漏洞，请发送电子邮件至 **[jaber@rightnowai.co](mailto:jaber@rightnowai.co)**。我们认真对待所有报告，并将在48小时内作出回应。

---

## 许可证

MIT。随心所欲地使用它。

---

## 链接

- [网站和文档](https://openfang.sh/)

- [快速入门指南](https://openfang.sh/docs/getting-started)

- [GitHub](https://github.com/RightNow-AI/openfang)

- [Discord 的中文翻译是“不和谐”或“纷争”。](https://discord.gg/sSJqgNnq6X)

- [推特/X](https://x.com/openfangg)

---

## 由 RightNow 建造

[RightNow 官网](https://www.rightnowai.co/)

OpenFang is built and maintained by [**Jaber**](https://x.com/Akashi203), Founder of [**RightNow**](https://www.rightnowai.co/).

[Website](https://www.rightnowai.co/) • [Twitter / X](https://x.com/Akashi203) • [![Buy Me A Coffee](https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png)](https://www.buymeacoffee.com/openfang)

---

**Built with Rust. Secured with 16 layers. Agents that actually work for you.**

## Reference

- [OpenFang - 代理操作系统](https://www.mcpworld.com/zh/detail/19af2411a031aec2315ee3724a6b10e7)
