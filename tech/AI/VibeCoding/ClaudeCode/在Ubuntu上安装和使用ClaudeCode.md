# Ubuntu上安装和使用ClaudeCode

## 1. 安装 Node.js

### 1.1 安装 npm

```bash
sudo apt install npm
```

安装完成后，Ubuntu 24.04 会安装一个 v18.19.1 版本的 nodejs。

### 1.2  安装 nvm

`NPM`（Node Version Manager）是最常用的 `node.js` 版本管理工具。它允许我们轻松安装、卸载和切换不同的 `node.js` 版本，并且支持跨平台使用。以下是 `NVM` 的安装和使用步骤:

### 1.2.1 安装 nvm

* 在 Linux 或 MacOS 上

可以通过以下命令安装 NVM：

```bash
sudo curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash
```

如果 GitHub 访问不了，或很慢，则使用镜像站，镜像站的站点可以来这里找：

```bash
https://freevaults.com/github-mirror-daily-updates/
```

例如，有一个 GitHub 镜像站是：[https://gitclone.com](https://gitclone.com) 。

```bash
sudo curl -o- https://gitclone.com/raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash
```

但是，这个镜像站好像只能镜像 github.com 这个域名，上面这个命令会告诉你路径不存在，所以只能用 clone 仓库的办法了。

或者，直接 clone 官方仓库：

```bash
sudo git clone https://github.com/nvm-sh/nvm.git
或
sudo git clone https://gitclone.com/github.com/nvm-sh/nvm.git
cd nvm
sudo ./install.sh
```

安装完成后, 需要重新加载配置文件让 nvm 生效：

```bash
sudo source ~/.bashrc
```

如果你用的是 zsh，则执行 `source ~/.zshrc`，或者直接关掉终端重新开一个也行。

用下列命令检查安装是否成功, 不成功可以调整一下环境变量的顺序:

```bash
sudo nvm -v
```

### 1.3 安装 node.js LTS版本

安装 nvm 后，就可以安装多个版本的 Node.js。例如，安装最新的 LTS 版本：

```bash
# 在 Linux 或 MacOS 上
sudo nvm install --lts

# 在 Windows 上使用下列命令
sudo nvm install lts
```

如果你需要特定版本，比如 v24.22.3，你可以通过以下命令安装：

```bash
sudo nvm install 24.22.3
```

### 1.4 切换 node.js 版本

查看已经安装的版本:

```bash
sudo nvm ls
```

可以使用 `nvm use` 命令切换 `node.js` 版本。例如，切换到 `node.js 24.22.3`：

```bash
sudo nvm use 24.22.3
```

此外，你还可以指定使用当前的 `LTS` 版本：

```bash
sudo nvm use lts
```

## 2. 更换 nvm 的安装源

### 2.1 安装 nrm

修改全局 .npmrc 文件（对所有项目有效）：

```bash
sudo vim ~/.npmrc

# 修改或添加如下内容
registry=https://registry.npmmirror.com/
```

### 2.2 npm 命令

通过命令行更改当前项目的源：

```bash
sudo npm config set registry https://registry.npmmirror.com/
```

查看当前源的命令:

```bash
sudo npm config get registry
```

### 2.3 nrm 包切换

安装依赖:

```bash
sudo npm install -g nrm
```

查看可用的源:

```bash
sudo nrm ls
```

```bash
$ sudo nrm ls
  npm ---------- https://registry.npmjs.org/
  yarn --------- https://registry.yarnpkg.com/
  tencent ------ https://mirrors.cloud.tencent.com/npm/
  cnpm --------- https://r.cnpmjs.org/
* taobao ------- https://registry.npmmirror.com/
  npmMirror ---- https://skimdb.npmjs.com/registry/
  huawei ------- https://repo.huaweicloud.com/repository/npm/
```

切换源:

```bash
sudo nrm use taobao
```

测试速度（所有源的响应时间）:

根据源的响应时间长短选择最快的源使用即可。

```bash
sudo nrm test
```

## 3. Claude Code

### 3.1 安装 Claude Code

```bash
sudo npm install -g @anthropic-ai/claude-code
```

日常使用：

```bash
cd {$YourProjectDir}    # 切换到你的项目目录
claude --dangerously-skip-permissions -c
```

### 3.2 cc-siwtch-cli

官网：[https://github.com/songunity/ccswitch-cli](https://github.com/songunity/ccswitch-cli)

CC-Swicth 有一个 CLI 的版本：

```bash
sudo npm install -g cc-switch-cli
```

第一次启动 cc-switch-cli，它会自动关闭 Claude Code 的官方登录，并且新增一个模型配置。

```bash
sudo cc-switch-cli
```

例如：

```bash
   CC Switch CLI v1.0.0
   ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─

   Providers (1):

 > * DeepSeek              sk-c5d...3534     anthropic   

   [Enter] Switch  [a] Add  [e] Edit  [d] Delete  [q] Quit
```

API-Key 可以使用 Shift + Ins 粘贴，粘贴完后，按 Tab 键就会显示完整的粘贴的 key，然后继续输入下一项。

### 3.3 Claude Code 配置文件

注：由于 `cc-switch-cli` 只能修改 root 用户的配置（即使你是使用非 root 用户登录的），所以如果是非 root 用户，只能自己手动修改配置文件了。

编辑 claude code 的配置文件：

```bash
sudo vim ~/.claude.json
```

可以看到，关闭 claude 官方登陆的设置项和其他：

```json
  "hasCompletedOnboarding": true,
```

供应商的配置文件：

```bash
sudo vim ~/.claude/settings.json
```

按照 DeepSeek 的官方映射规则，如果模型名以 `claude-opus` 开头，请求确实会被路由到更贵的 `deepseek-v4-pro`，并按 V4 Pro 的价格计费。要确保只使用 `deepseek-flash`，关键在于彻底避开 `claude-opus` 这个前缀。

调整方案：把 Opus 槽位也指向 Flash
既然你已经理解了不能随意填 `claude-opus-*`，那么解决办法就是：把 Claude Code 里所有能配置模型的变量，尤其是 `ANTHROPIC_DEFAULT_FABLE_MODEL`、 `ANTHROPIC_DEFAULT_OPUS_MODEL`，都明确指向 `claude-sonnet` 或 `claude-haiku` 这类会被映射到 `deepseek-flash` 的名称。

详细可以参考：[DeepSeek-v4.1-flash 对话历史](https://chat.deepseek.com/share/8s6qg9l7w37xurq6ze)

所以，添加如下的配置，例如：

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "https://api.deepseek.com/anthropic",
    "ANTHROPIC_AUTH_TOKEN": "你的DeepSeek API密钥",
    "ANTHROPIC_MODEL": "claude-sonnet-5",
    "ANTHROPIC_DEFAULT_FABLE_MODEL": "claude-sonnet-5",
    "ANTHROPIC_DEFAULT_OPUS_MODEL": "claude-sonnet-5",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "claude-sonnet-5",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "claude-haiku-5"
  }
}
```
