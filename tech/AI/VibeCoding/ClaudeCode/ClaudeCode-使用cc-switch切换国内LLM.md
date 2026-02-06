# ClaudeCode 中使用 cc-switch 切换国内LLM

## 概述

cc-switch 是一个便捷的工具，可以快速切换 Claude Code 的 API 配置。

## 1. 安装 cc-switch

Windows:

前往 [cc-switch GitHub Releases](https://github.com/farion1231/cc-switch/releases) 页面下载最新版本的安装包。

MacOS / Linux:

```bash
brew tap farion1231/ccswitch
brew install --cask cc-switch
brew upgrade --cask cc-switch
```

## 2. 添加 MiniMax 配置

启动 cc-switch，点击右上角 ”+” ，选择预设的 MiniMax 供应商，并填写您的 MiniMax API Key。

## 3. 配置模型名称

将模型名称全部改为 MiniMax-M2.1，完成后点击右下角的 “添加”。

base url 的值是：

```bash
ANTHROPIC_BASE_URL = https://api.minimaxi.com/anthropic
```

国际用户使用 https://api.minimax.io/anthropic，中国大陆用户使用 https://api.minimaxi.com/anthropic 。

API 格式是：Anthropic Message(原生) 。

## 4. 启用配置

回到首页，点击 “启用” 即可开始使用。

## 5. 编辑配置文件

编辑或新增 .claude.json 文件，MacOS & Linux 为 ~/.claude.json，Windows 为用户目录 /.claude.json 。

```bash
# 在 json 文件头部或尾部，新增 `hasCompletedOnboarding` 参数
{
  "hasCompletedOnboarding": true,

  ..............
  ... 原来的 json 内容 ...
  ..............
}
```

## 6. 启动 Claude Code

配置完成后，进入工作目录，在终端中运行 claude 命令以启动 Claude Code 。

## 7. 信任文件夹

启动后，选择 信任此文件夹 (Trust This Folder)，以允许 Claude Code 访问该文件夹中的文件，随后开始在 Claude Code 中使用 MiniMax-M2.1 。

## 8. 参考文章

- [在 Claude Code 中使用 MiniMax-M2.1（推荐）](https://platform.minimaxi.com/docs/guides/text-ai-coding-tools)
