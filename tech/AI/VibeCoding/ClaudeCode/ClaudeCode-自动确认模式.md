# ClaudeCode 自动确认模式

## 1. 概述

Claude Code 按 Ctrl + Tab 可以切换到 accept edit on 模式，这种模式对文件的所有编辑将自动应用，无需你手动确认，但执行 Bash 命令的时候需要你手动确认。

## 2. 自动确认模式

有两种方式：

### 2.1 程序化执行模式 (Agent SDK)

这是实现命令执行自动确认的直接方法，可以在脚本或自动化流程中使用。

* **基础用法**：

  通过 -p（或 --print）参数运行命令，Claude 将非交互式地执行任务，所有操作均不会等待手动确认。

  ```bash
  claude -p "你的任务描述"
  ```

* **关键**：

  * 指定允许的工具**：为了实现自动化，必须通过 --allowedTools 参数明确授权 Claude 可以自动使用哪些工具。权限规则语法很灵活，例如：

    * --allowedTools "Bash"：允许运行所有 Bash 命令。

    * --allowedTools "Bash(git diff:*)"：允许运行以 git diff 开头的任何命令（:* 表示前缀匹配）。

    * --allowedTools "Bash,Read,Edit"：允许执行 Bash 命令、读取和编辑文件，形成一个完整的自动化工作流。

* **实际应用示例**：

  * **自动运行并修复测试**：以下命令授权 Claude 自动执行测试、读取文件（查看错误）、编辑文件（修复代码）而不请求许可。

  ```bash
  claude -p "运行测试套件并修复所有失败项" --allowedTools "Bash,Read,Edit"
  ```

  * **自动创建提交**：以下命令授权Claude使用相关的git命令来查看状态和创建提交。

  ```bash
  claude -p "查看我的暂存更改并创建一个合适的提交" --allowedTools "Bash(git diff:*),Bash(git log:*),Bash(git status:*),Bash(git commit:*)"
  ```

  注意：这种模式是后台执行的，看不到交互界面，所以如果 LLM 会要你回答一些选择才能继续的任务，是不适合的，只适合那种全自动，日常的后台任务，推荐用第二种方式。

### 2.2 bypass permissions 模式

`--dangerously-skip-permissions` 模式。

```bash
claude --dangerously-skip-permissions
```

打开 bypass permissions on 模式。

系统配置文件 `~/.claude.json 里会看到多了一个设置，如下所示：

```json
{
  "bypassPermissionsModeAccepted": true
}
```

这种方式可以交互，非常推荐，但是 Claude Code 完成一个阶段的任务还是会停下来，所以尽量安排它做更长的任务，如果可以的话。

### 2.3 自动化操作的安全提示

在使用自动化模式前，请务必了解以下风险：

- 评估风险：自动确认意味着Claude获得了很高的操作权限。请务必在受控环境（如功能分支、测试项目）中先行测试。

- 使用检查点：对于复杂的操作，可以要求Claude先提供一个“计划”，或在执行前询问“安全影响”，利用其内置的安全检查。

如果你想了解如何将这种自动化集成到持续集成（CI）流程中，或者有其他特定场景需要规划，我可以提供进一步的建议。

## 3. 参考文章

- [DeepSeek v3.2：Claude Code里如何自动确认命令行命令的执行？](https://chat.deepseek.com/a/chat/s/5a27f063-1b2d-499d-bb9d-7f6d5cf6e412)

- [玩转 Claude Code 的 23 个实用小技巧](https://zhipu-ai.feishu.cn/wiki/LhMdwIWmrihNNTkp69xc4syTnUe)
