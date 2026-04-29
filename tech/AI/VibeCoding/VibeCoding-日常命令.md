# Vibe Coding 常用命令 

## CLI

1. Claude Code

安装与升级：

```bash
npm install -g @anthropic-ai/claude-code
```

日常使用：

```bash
cd {$YourProjectDir}    # 切换到你的项目目录
claude -c --dangerously-skip-permissions
```

2. Codex

安装与升级：

```bash
npm install -g @openai/codex
```

日常使用：

```bash
cd {$YourProjectDir}    # 切换到你的项目目录
codex --dangerously-bypass-approvals-and-sandbox
```

进入以后用 /resume 命令恢复会话

3. OpenCode

安装与升级：

```bash
npm install -g opencode-ai
```

日常使用：

```bash
cd {$YourProjectDir}    # 切换到你的项目目录
opencode -c
```

