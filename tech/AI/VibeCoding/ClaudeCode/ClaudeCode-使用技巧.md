# ClaudeCode 使用技巧

## 1. Superpowers

官网：[https://github.com/obra/superpowers](https://github.com/obra/superpowers)

### 1.1 安装 Superpowers

在 Claude Code 中，先注册 marketplace ：

```bash
/plugin marketplace add obra/superpowers-marketplace
```

然后在 marketplace 里安装 plugin ：

```bash
/plugin install superpowers@superpowers-marketplace
```

Note: 不同的平台的安装有所不同，Claude Code 有一个内建的 plugin 系统，而 Codex，OpenCode 则需求手动安装。

验证安装

开始一个 Claude session 向 LLM 提问来触发 skill (例如："help me plan this feature" or "let's debug this issue"). Claude 将自动调用 superpowers skill.

