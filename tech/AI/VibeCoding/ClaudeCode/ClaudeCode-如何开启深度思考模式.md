# Claude Code 如何开启深度思考模式

## 方法

Claude Code 如何开启深度思考模式？

1. 编辑 `Claude Code` 的配置文件 `~/.claude/settings.json`，在 env 中新增以下内容：

    ```json
      "CLAUDE_CODE_EXTRA_BODY": {
          "thinking": {
              "type": "enabled"
          }
      }
    ```

2. 配置完成后执行命令 `claude` 重新启动。
