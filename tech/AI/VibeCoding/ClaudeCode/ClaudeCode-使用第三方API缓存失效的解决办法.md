# Claude Code 使用第三方 API 缓存失效的解决办法

## 起因

从 2026 年 2 月左右，如果你的 `Claude Code` 配置的是第三方 API，用的时候发现推理变慢，token 暴涨，很可能不是第三方 API 的问题。

这是因为，`Anthropic` 在 `Claude Code` 每次请求的上下文的 `System Prompt`（系统提示词）中的第一句里，加了这样一段（偷塞了 5 个可变的字符）：

```text
x-anthropic-billing-header; cc_version=2.1.143.f09; cc_entrypoint=cli; cch=0f646;
```

其中 `cch=xxxxx` 的值每次会话都会变，这会导致使用第三方 API 时，缓存命中降低，而 `Anthropic` 自己处理的时候是把这一段内容忽略掉的，所以对他自己没有影响。

这是官方故意设计的防御机制。如果你模仿 Claude Code 做了一个 Coding Agent，上下文的所有格式和数据都是官方是一样的，但是你的 `cch` 值是不会变的，如果你调用官方的 API，他立刻就能识别出来，从而拒绝服务。原因是这样的：

从泄露的源代码可以看到，源码里只组合了上面那段字符串，cch 的默认值是 `cch=00000` ，在源码中并未做任何修改。真正的修改是在 `Bun` 里编译出来的 `claude` 执行文件里，`Claude Code` 是使用 `Bun` 来编译的 `Node.js` 代码，而 `Bun` 是用 `Zig` 写的一个 `Node.js` 编译和执行器。`Anthropic` Fork 了官方的 `Bun` ，并做了 hack，在输出 respone 到 HTTP 时，偷偷把 `cch=00000` 中的 `00000` 替换为真正的 cch 哈希值，最终输出。而官方的 `Bun` 版本是没有这个步骤的。

目前已证实 `DeepSeek V4 pro` 的 API 调用是能识别这个防盗用策略的，所以缓存命中率很高，不受其影响。但如果你所使用的第三方 API 没有对这段信息特殊处理，则会导致大部分时间里缓存命中率都很低，从而导致 token 扣费变高，推理速度变慢。

`Claude Code` 的上下文的顺序是：

`MCP 定义` -> `Tools 定义` -> `Skills 定义` -> `System Prompt` -> `User Message`

这会导致缓存只能命中到 `MCP 定义` -> `Tools 定义` -> `Skills 定义`，后面的都不能命中，导致命中率变得很低。

## 解决办法

解决的办法就是，在 `~/.claude/settings.json` 用户配置文件里添加如下内容：

```json
{
  ......,

  "env": {
    "ANTHROPIC_BASE_URL": "https://token-plan-cn.xiaomimimo.com/anthropic",
    "ANTHROPIC_AUTH_TOKEN": "",
    "ANTHROPIC_MODEL": "mimo-v2.5-pro[1m]",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "mimo-v2.5[1m]",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "mimo-v2.5-pro[1m]",
    "ANTHROPIC_DEFAULT_OPUS_MODEL": "mimo-v2.5-pro[1m]",
    "ANTHROPIC_REASONING_MODEL": "mimo-v2.5-pro[1m]",
    "CLAUDE_CODE_SUBAGENT_MODEL": "mimo-v2.5-pro[1m]",
    "CLAUDE_CODE_EFFORT_LEVEL": "high",

    "CLAUDE_CODE_ATTRIBBUTTON_HEADER": "0"
  }
}
```

添加的内容是：`"CLAUDE_CODE_ATTRIBBUTTON_HEADER": "0"` 。

加上这个设置后，退出 Claude Code，再重启，就不会再用 `x-anthropic-billing-header; cc_version=2.1.143.f09; cc_entrypoint=cli; cch=0f646;` 这段内容了，也就不会影响缓存命中率了。

## 参考视频

- [https://www.bilibili.com/video/BV1m2LG6WEdH](https://www.bilibili.com/video/BV1m2LG6WEdH)
