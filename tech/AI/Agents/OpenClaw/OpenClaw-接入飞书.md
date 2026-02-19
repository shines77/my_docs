# OpenClaw 接入飞书

## 飞书插件

官网：[https://github.com/m1heng/clawdbot-feishu](https://github.com/m1heng/clawdbot-feishu)

## 安装

```bash
# 先切换到 npm 的全局安装目录
cd C:\Users\shines77\AppData\Roaming\npm

# 然后再执行安装
openclaw plugins install @m1heng-clawd/feishu
```

## 升级

```bash
openclaw plugins update feishu
```

## 启用插件

```bash
# 先把 C:\Users\shines77\AppData\Roaming\npm\node_modules\openclaw\node_modules\@m1heng-clawd 下的 feishu 目录拷贝一份到 C:\Users\shines77\AppData\Roaming\npm\node_modules\openclaw\extensions 下面

xcopy C:\Users\shines77\AppData\Roaming\npm\node_modules\openclaw\node_modules\@m1heng-clawd\feishu C:\Users\shines77\AppData\Roaming\npm\node_modules\openclaw\extensions\feishu /E /I /H /Y

# 再启用它
openclaw plugins enable feishu
```

启用成功后，使用下面命令查看已安装的插件列表：

```bash
openclaw plugins list
```

## 问题

### 1. 子进程报错

如果安装的时候报错：

```bash
Downloading @m1heng-clawd/feishu…
[openclaw] Failed to start CLI: Error: spawn EINVAL
    at ChildProcess.spawn (node:internal/child_process:420:11)
    at spawn (node:child_process:796:9)
    at runCommandWithTimeout (file:///C:/Users/shines77/AppData/Roaming/npm/node_modules/openclaw/dist/process/exec.js:83:19)
    at installPluginFromNpmSpec (file:///C:/Users/shines77/AppData/Roaming/npm/node_modules/openclaw/dist/plugins/install.js:306:23)
    at async Command.<anonymous> (file:///C:/Users/shines77/AppData/Roaming/npm/node_modules/openclaw/dist/cli/plugins-cli.js:367:24)
    at async Command.parseAsync (C:\Users\shines77\AppData\Roaming\npm\node_modules\openclaw\node_modules\commander\lib\command.js:1122:5)
    at async runCli (file:///C:/Users/shines77/AppData/Roaming/npm/node_modules/openclaw/dist/cli/run-main.js:59:5)
```

不要尝试全局安装：

```bash
npm install -g @m1heng-clawd/feishu
```

虽然全局安装不会报错，但是在启用 feishu 插件时，依然会报错，见下，解决方法见最后。

### 2. 启用 feishu 插件时报错

在执行启用命令时：

```bash
openclaw plugins enable @m1heng-clawd/feishu
```

报如下错误：

```bash
[openclaw] Failed to start CLI: Error: Config validation failed: plugins.entries.@m1heng-clawd/feishu: plugin not found: @m1heng-clawd/feishu
    at Object.writeConfigFile (file:///C:/Users/shines77/AppData/Roaming/npm/node_modules/openclaw/dist/config/io.js:395:19)
    at writeConfigFile (file:///C:/Users/shines77/AppData/Roaming/npm/node_modules/openclaw/dist/config/io.js:504:28)
    at Command.<anonymous> (file:///C:/Users/shines77/AppData/Roaming/npm/node_modules/openclaw/dist/cli/plugins-cli.js:233:15)
    at Command.listener [as _actionHandler] (C:\Users\shines77\AppData\Roaming\npm\node_modules\openclaw\node_modules\commander\lib\command.js:568:17)
    at C:\Users\shines77\AppData\Roaming\npm\node_modules\openclaw\node_modules\commander\lib\command.js:1604:14
    at C:\Users\shines77\AppData\Roaming\npm\node_modules\openclaw\node_modules\commander\lib\command.js:1485:33
    at async Command.parseAsync (C:\Users\shines77\AppData\Roaming\npm\node_modules\openclaw\node_modules\commander\lib\command.js:1122:5)
    at async runCli (file:///C:/Users/shines77/AppData/Roaming/npm/node_modules/openclaw/dist/cli/run-main.js:59:5)
```

这是因为找不到 feishu 的目录在哪，其实是要安装到 C:\Users\shines77\AppData\Roaming\npm\node_modules\openclaw\extensions 下面，且不能带 @m1heng-clawd 命名空间，这就可能需要自己手动复制一下。

你的插件通常存放在以下两个位置之一：

1. 全局配置与插件目录 (最可能的位置)

    OpenClaw 默认会将下载的插件和工作环境放在当前用户的根目录下：

    C:\Users\shines77\.openclaw\plugins\node_modules

    在这个目录下，你应该能看到 @m1heng-clawd 文件夹。

2. **npm** 全局隔离目录

    如果你是通过 `npm install -g openclaw` 全局安装的 OpenClaw，且 OpenClaw 没能创建用户级目录，它可能会尝试链接到：

    %AppData%\npm\node_modules\openclaw (即：C:\Users\shines77\AppData\Roaming\npm\node_modules\openclaw)

因为我们是使用全局安装的 OpenClaw，所以我们要切换到第二个位置的目录去执行安装命令，再启用它。

```bash
cd C:\Users\shines77\AppData\Roaming\npm\node_modules\openclaw

# 在全局 npm 的 \openclaw 目录下安装 (不要用全局安装)
npm install @m1heng-clawd/feishu

# 然后把 C:\Users\shines77\AppData\Roaming\npm\node_modules\openclaw\node_modules\@m1heng-clawd 下的 feishu 目录拷贝一份到 C:\Users\shines77\AppData\Roaming\npm\node_modules\openclaw\extensions 下面

xcopy C:\Users\shines77\AppData\Roaming\npm\node_modules\openclaw\node_modules\@m1heng-clawd\feishu C:\Users\shines77\AppData\Roaming\npm\node_modules\openclaw\extensions\feishu /E /I /H /Y

# 再启用它，此时不用使用 @m1heng-clawd/，直接写 feishu 即可。
openclaw plugins enable feishu
```

看到如下信息就是成功了。

```bash
openclaw plugins enable feishu

Enabled plugin "feishu". Restart the gateway to apply.
```

需要 restart 一次 gateway 才能生效。

## 创建机器人

方法见：[百度云：OpenClaw接入飞书](https://cloud.baidu.com/doc/LS/s/Wml21yd8l)

## 配置飞书插件

### 设置 feishu channel

先添加飞书的 channel：

```bash
openclaw channels add --channel feishu --token "{Your_appId}:{Your_appSecret}"
```

例如：

```bash
openclaw channels add --channel feishu --token "Abcdefghjiklmn:HUDYDHS:LSSDKS"
```

再设置具体参数，配置文件在 `~/.openclaw/openclaw.json`：

```bash
vim ~/.openclaw/openclaw.json
```

在合适的位置插件如下内容

```json
  "channels": {
    "feishu": {
      "appId": "Your appId",
      "appSecret": "Your appSecret",
      "domain": "feishu",
      "groupPolicy": "allowlist"
    }
  }
```

如果只是设置其中一个值，可以使用如下的格式：

```bash
openclaw config set XXXXX.YYYY ZZZZ

# 例如：
openclaw config set channels.feishu.appId "Your appId"
```

最后，通过下列命令查看配置文件。

```bash
openclaw config edit
```

然后重启 gateway ：

```bash
openclaw gateway restart
```

## 配置飞书后台

步骤一：回到飞书开发者后台页面，在 `应用管理页` 左侧导航栏找到 `事件与回调`，设置 `事件配置` 和 `回调配置` 的 `订阅方式` 均为使用 `长连接模式` 。

步骤二：点击页面中的 “添加事件”，在弹出的事件列表中，选择 `消息与群组` 分类，分别勾选其中的 `接收消息`，`消息已读`，`用户进入和机器人对话`，`用户和机器人的会话首次被创建` 等 4 个事件，点击 “确定添加”，“确定开通权限”，完成事件订阅。

注意：如果这一步提示 `未建立长连接`，请检查自己的 `APP ID` 和 `APP Secret` 是否已正确配置。

步骤三：配置飞书应用权限，打开在 `应用管理页左侧导航栏` 找到 `权限管理` 。

把下面的内容复制进去：

```bash
{
  "scopes": {
    "tenant": [
      "contact:user.base:readonly",
      "im:chat",
      "im:chat:read",
      "im:chat:update",
      "im:message",
      "im:message.group_at_msg:readonly",
      "im:message.p2p_msg:readonly",
      "im:message:send_as_bot",
      "im:resource"
    ],
    "user": []
  }
}
```

步骤四：发布刚才编辑好的应用，发一个新版本，这里应该算是 1.0.1 版本了。

步骤五：验证飞书接入。打开手机或电脑端飞书 APP，登录与飞书开发者平台相同的账号，找到工作台入口点击进入，已发布的应用 OpenClaw 助手，点击进入即可进行对话。
