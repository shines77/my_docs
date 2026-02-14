# ClaudeCode TeamMate 模式

## 1. 命令行

开启 TeamMate 模式，同时开启自动驾驶模式（无需手动确认）：

```bash
claude --dangerously-skip-permissions --teammate-mode tmux
```

注：tmux 是 Unix/Linux 下的终端分屏工具，Windows 下用不了。

## 2. 配置文件

编辑 `~/.claude.json` 文件，添加如下设置：

```json
{
  ..........
  // 其他设置
  ..........

  "env": {
    "CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS": "1"
  },
  "teammateMode": "auto"
}
```
