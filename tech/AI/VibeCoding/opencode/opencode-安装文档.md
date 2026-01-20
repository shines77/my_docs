# OpenCode 安装文档

## 1. 四种形式

* OpenCode CLI
* IDE客户端
* IDE差距
* 云端

本文只介绍 OpenCode CLI 的安装。

## 2. 安装 OpenCode CLI

安装 OpenCode Terminal 。

### 2.1 MacOS / Linux 用户

```bash
curl -fsSL https://opencode.ai/install -o install.sh
chmod +x ./install.sh

./install.sh
```

### 2.2 Windows 用户

如果你是 Windows 用户，需要在 Node.js 环境下运行：

```bash
npm i -g opencode-ai
```

如果你有 bun，则直接运行：

```bash
bun add -g opencode-ai
```

如果你有 brew，则直接运行：

```bash
brew install anomalyco/tap/opencode
```

如果你有 paru，则直接运行：

```bash
paru -S opencode
```

### 2.3 检查安装是否成功

运行：

```bash
opencode-ai --version
```

如果能看到版本号即代表安装成功，例如这里会显示：`1.1.27`。

## 3. 配置 OpenCode

OpenCode 的一大优势是 模型可配置。

你可以选择：

* 本地模型（如 Ollama）
* 远程 API（如 OpenAI、其他兼容接口）

常见配置步骤包括：

* 设置模型名称
* 配置 API Key
* 指定运行方式

完成后，OpenCode 就知道该“用哪个大脑”了。

## 4. 开始使用 OpenCode

进入你的代码目录：

```bash
cd your-project
opencode
````

你现在可以直接输入指令，例如：

* “解释这个项目的结构”
* “帮我重构这个函数”
* “这个报错是什么意思？”

OpenCode 会结合当前代码上下文给出回答。

## 5. OpenCode 命令

在 CLI 的状态下，使用如下命令：

* `/models`：设置使用的大模型版本。

在命令行的模式下使用下列命令可以查看所有默认的模型：

```bash
opencode models
```

默认的模型列表：

```bash
opencode/big-pickle
opencode/glm-4.7-free
opencode/gpt-5-nano
opencode/grok-code
opencode/minimax-m2.1-free
```

serve - 启动无头 HTTP API 服务器

Start a headless OpenCode server for API access.

```bash
opencode serve
```

Flags

| Flag         | Description                                |
| :------      | :----------                                |
| `--port      | Port to listen on                          |
| `--hostname  | Hostname to listen on                      |
| `--mdns      | Enable mDNS discovery                      |
| `--cors      | Additional browser origin(s) to allow CORS |

web - 启动带网页界面的服务器

Start a headless OpenCode server with a web interface.

```bash
opencode web
```

Flags

| Flag         | Description                                |
| :------      | :----------                                |
| `--port`     | Port to listen on                          |
| `--hostname` | Hostname to listen on                      |
| `--mdns`     | Enable mDNS discovery                      |
| `--cors`     | Additional browser origin(s) to allow CORS |

## 参考文章

1. [OpenCode Docs](https://opencode.ai/docs/cli/)

2. [OpenCode Docs 中文版](https://opencodeguide.com/zh/docs/cli)

3. [《全网最全！OpenCode 安装使用保姆级教学，彻底告别付费 Claude Code！》](https://zhuanlan.zhihu.com/p/1992972881009856726)
