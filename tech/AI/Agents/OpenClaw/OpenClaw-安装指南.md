# OpenClaw 安装指南

## 1. 概述

`OpenClaw`（曾用名 `ClawdBot`）是一个运行在你个人电脑或服务器上，能够通过聊天软件（如 `WhatsApp`、`Telegram`、`iMessage`）接收指令并实际操控你的电脑系统（如操作文件、浏览器、执行命令）来完成任务的开源 AI 智能体。`Claw` 是龙虾的意思，所有也俗称为“大龙虾”。

### 1.1 OpenClaw **是什么**

`OpenClaw` 是一个能 "指挥" 计算机的全天候数字员工 (`Agent`)，通过连接通讯软件和大模型 API，实现对电脑的自动化操作。基于 `OpenClaw` 框架提供以下核心功能：

- 文件操作：读写/编辑文件（read/write/edit），管理项目文档。
- 终端执行：运行 Shell 命令（exec），支持后台任务与交互式 CLI。
- 内存管理：通过 MEMORY.md 和 memory/YYYY-MM-DD.md 持续学习上下文。
- 跨会话协作：通过 sessions_spawn 调度子代理处理复杂任务。

## 2. 安装 OpenClaw

官网：[https://openclaw.ai/](https://openclaw.ai/)

推荐在图形界面下安装 OpenClaw，这样可玩性更高。如果在非图形界面的服务器上安装，又想在外网用网页端来使用 OpenClaw，则需要在你的服务器上配置 https，否则 http 链接只能看不能使用，这样是为了防止个人信息泄露。而没有图形界面，在服务器的本地环境是无法浏览网页的。

本文是在 Windows 下安装配置的， OpenClaw 的版本是：`OpenClaw 2026.2.1`。

安装有很多种方法：

### 2.1 命令行

#### 2.1.1 Windows

注意：使用 CMD 或者 PowerShell，最好以管理员身份运行。先把 CMD 和 PowerShell 固定到开始“屏幕”，然后在开始菜单里，它们的图标上鼠标右键，菜单里选“以管理员身份运行”，即可，下同，不早赘述。

CMD 环境下：

```bash
# Works everywhere. Installs everything. You're welcome.
curl -fsSL https://openclaw.ai/install.cmd -o install.cmd && install.cmd && del install.cmd
```

PowerShell 环境下：

以管理员身份运行 PowerShell，方法见上。

先执行这两条命令：

```powershell
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

这是什么意思？

- 第一条命令：允许当前用户运行本地和下载的脚本
- 第二条命令：允许当前用户运行本地和下载的脚本

> 安全提示：这些命令只会影响您自己的账户，不会影响系统安全或其他用户。

执行安装命令：

```powershell
# Works everywhere. Installs everything. You're welcome.
iwr -useb https://openclaw.ai/install.ps1 | iex
```

安装过程会自动完成：

- 检测系统环境
- 安装必要依赖（Node.js 等）
- 下载 OpenClaw 核心文件
- 配置环境变量
- 启动配置向导

#### 2.1.2 Linux/MacOS

```bash
# Works everywhere. Installs everything. You're welcome.
curl -fsSL https://openclaw.ai/install.sh | bash
```

### 2.2 npm 安装（node.js）

需要先安装 node.js，版本要求 ≥22.12.0，推荐安装官方推荐的最新 LTS 版本，例如：[Windows x64 - v24.13.0](https://nodejs.org/dist/v24.13.0/node-v24.13.0-x64.msi)。

官网(需要科学上网)：[https://nodejs.org](https://nodejs.org)<br/>
Nodejs中文网(非官方)：[https://node.org.cn/en](https://node.org.cn/en)<br/>
Node.js 中文文档：[https://www.nodeapp.cn/](https://www.nodeapp.cn/)<br/>

如果默认的安装源访问不了，可以换国内的源：

```bash
# 设置为淘宝镜像源（推荐国内用户）
npm config set registry https://registry.npmmirror.com/
# 设置为腾讯云镜像源
npm config set registry https://mirrors.cloud.tencent.com/npm/
# 设置为华为云镜像源
npm config set registry https://repo.huaweicloud.com/repository/npm/
# 恢复官方源
npm config set registry https://registry.npmjs.org/
```

安装 OpenClaw ：

```bash
# Install OpenClaw
npm i -g openclaw@latest

# Config your openclaw
openclaw onboard
```

### 2.3 hackable

如果你想，可以下载源码来自己改。

```bash
# For those who read source code for fun
curl -fsSL https://openclaw.ai/install.sh | bash -s -- --install-method git
```

## 3. 配置 OpenClaw

启动配置（初始化 OpenClaw）：

```bash
openclaw onboard
```

这会启动一个交互式终端界面，引导你完成：

1. 确认继续：这一步主要是告诉你，使用 OpenClaw 可能会有一些风险。请问你是否继续？选 Yes，然后回车。

2. 选择模式：新手选 QuickStart，老手选 Advanced，默认选 QuickStart 即可。

3. 配置模型：选择 LLM API 提供商，填入 API Key，这里推荐选 QWen，选择并确定后，下一步选择 QWen OAuth （网页验证），确定选择以后，会给你一个验证的网址，打开网页去注册和验证一下就可以了。因为目前是免费的，登录网页注册和认证一下就可以了，注意这个是国际版的 QWen，不是国内版，所以你可能需要注册一下账号。

    验证完成后，会让你选具体的模型，例如：

    ```bash
    ◆  Default model
    │  ● Keep current (qwen-portal/coder-model)
    │  ○ Enter model manually
    │  ○ qwen-portal/coder-model
    │  ○ qwen-portal/vision-model
    ```

    选第一个就可以了，这个是编程用的大模型，也够用了，就两个可选的模型。

4. 设置渠道：选择聊天应用（Telegram、Discord 等），默认是不支持飞书、钉钉、QQ、企业微信的，要自己安插件才能支持，这里选最后一个 “Skip for now”，先跳过。

5. 设置 skills：这里也可以先选 no 跳过。之后再装 skills 。

6. 启用 hooks：建议三个都勾上。

7. 然后启动 OpenClaw。

8. Install shell completion script? 是否安装在 Shell 中使用 OpenClaw 的自动补全功能。

然后在浏览器输入如下网址访问，例如：

如果 127.0.0.1 访问不了，改成 localhost 试试。

```bash
http://localhost:18789/?token=561529b6efeb7eca0bcfaf92fcc0b70978133e729d54821a
```

输入：“你好，你是什么大模型？”

回答：

```bash
我是 Qwen，由通义实验室开发的大语言模型。我被用作 OpenClaw 系统的推理引擎。

不过，与其关注我是什么模型，不如告诉我你需要完成什么任务？我可以帮你处理各种工作，比如：

编写代码、文档或邮件
分析数据或文件内容
自动化系统操作
搜索网络信息
管理日程或提醒事项
控制智能家居设备
或者任何其他需要帮助的事情
直接告诉我你的需求即可，我会尽力协助你。
```

```bash
# Install gateway service (后台运行 gateway)
openclaw onboard --install-daemon

# Open Terminal UI
openclaw tui

# Open the Control UI with your current token
openclaw dashboard

# Start the gateway
openclaw gateway start
openclaw gateway restart
```

## OpenClaw 的升级

如果你不是用 git 仓库导出，从源码安装的话，是不能使用 update 命令来升级的。

```bash
openclaw update
```

如果你用的是 npm 安装的话，必须用你的包管理来升级 openclaw。

```bash
npm i -g openclaw@latest
```

## 安装 Chrome 浏览器扩展

安装 Chrome 扩展：

```bash
openclaw browser extension install

C:\Users\shines77\.openclaw\browser\chrome-extension
```

会显示安装的路径，并得到下列的提示：

```bash
Next:

- Chrome → chrome://extensions → enable “Developer mode”
- “Load unpacked” → select: C:\Users\shines77\.openclaw\browser\chrome-extension
- Pin “OpenClaw Browser Relay”, then click it on the tab (badge shows ON)
```

- 先在 Chrome 里输入 `chrome://extensions`，并打开 "开发者模式"，在屏幕的右上角。
- 点击 “加载未打包的扩展程序” 按钮（即手动安装扩展），选择刚才安装的扩展的路径：C:\Users\shines77\.openclaw\browser\chrome-extension 。
- 会显示一个叫 "OpenClaw Browser Relay" 的扩展。
- 然后进入扩展管理界面，找到 "OpenClaw Browser Relay"，选项里把它钉（pin）在工具栏上，然后在 Tab 选项栏上点击它的图标。

```bash
# 如果是 Windows 下的 PowerShell
openclaw browser create-profile `
  --name my-chrome `
  --driver extension `
  --cdp-url http://localhost:18792 `
  --color "#00AA00"

# 如果是 Linux 则
openclaw browser create-profile \
  --name my-chrome \
  --driver extension \
  --cdp-url http://localhost:18792 \
  --color "#00AA00"
```

```bash
openclaw browser --browser-profile my-chrome tabs
```

- 然后在浏览器里打开：

```bash
http://localhost:18792/
```

如果显示 OK。则表示启动成功。

[canvas] mount 点：

```bash
http://localhost:18789/__openclaw__/canvas/
```

## 参考文章

- [OpenClaw(Clawd/Moltbot)汉化版搭建指南](https://cloud.tencent.com/developer/article/2626168)

- [OpenClaw 中文汉化版](https://openclaw.qt.cool/)
