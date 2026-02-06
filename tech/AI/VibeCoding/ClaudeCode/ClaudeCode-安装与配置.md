# ClaudeCode 安装与配置

## 1. 安装 Claude Code

### 1.1 命令行安装

macOS, Linux, WSL:

```bash
curl -fsSL https://claude.ai/install.sh | bash
```

Windows PowerShell:

```bash
irm https://claude.ai/install.ps1 | iex
```

Windows CMD:

```bash
curl -fsSL https://claude.ai/install.cmd -o install.cmd && install.cmd && del install.cmd
```

（注：Windows 上需要以管理员身份运行 PowerShell 或 CMD 。）

安装完成后，验证一下:

```bash
claude --version
```

如果显示版本号，说明安装成功！

> Native installations automatically update in the background to keep you on the latest version.

### 1.2 npm 安装

你的电脑需要安装了 Node.js，在命令行输入：

```bash
node --version
```

如果显示版本号(比如 v18.17.0)，说明已安装。如果没有，去 nodejs.org 下载安装。

（注：Windows 上需要以管理员身份运行 PowerShell 或 CMD 。）

打开命令行，输入以下命令:

```bash
npm install -g @anthropic-ai/claude-code
```

等待安装完成(可能需要几分钟)。

安装完成后，验证一下:

```bash
claude --version
```

如果显示版本号，说明安装成功！