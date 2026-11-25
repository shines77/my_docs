# Ubuntu上安装和使用ClaudeCode

## 1. 安装 Node.js

### 1.1 安装 npm

```bash
apt install npm
```

安装完成后，Ubuntu 24.04 会安装一个 v18.19.1 版本的 nodejs。

### 1.2  安装 nvm

`NPM`（Node Version Manager）是最常用的 `node.js` 版本管理工具。它允许我们轻松安装、卸载和切换不同的 `node.js` 版本，并且支持跨平台使用。以下是 `NVM` 的安装和使用步骤:

### 1.2.1 安装 nvm

* 在 Linux 或 MacOS 上

可以通过以下命令安装 NVM：

```bash
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash
```

如果 GitHub 访问不了，或很慢，则使用镜像站，镜像站的站点可以来这里找：

```bash
https://freevaults.com/github-mirror-daily-updates/
```

例如，有一个 GitHub 镜像站是：[https://gitclone.com](https://gitclone.com) 。

```bash
curl -o- https://gitclone.com/raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash
```

或者，直接 clone 官方仓库：

```bash
git clone https://github.com/nvm-sh/nvm.git
cd nvm
./install.sh
```

安装完成后, 需要重新加载配置文件让 nvm 生效：

```bash
source ~/.bashrc
```

如果你用的是 zsh，则执行 `source ~/.zshrc`，或者直接关掉终端重新开一个也行。

用下列命令检查安装是否成功, 不成功可以调整一下环境变量的顺序:

```bash
nvm -v
```

### 1.3 安装 node.js LTS版本

安装 nvm 后，就可以安装多个版本的 Node.js。例如，安装最新的 LTS 版本：

```bash
# 在 Linux 或 MacOS 上
nvm install --lts

# 在 Windows 上使用下列命令
nvm install lts
```

如果你需要特定版本，比如 v24.22.3，你可以通过以下命令安装：

```bash
nvm install 24.22.3
```

### 1.4 切换 node.js 版本

查看已经安装的版本:

```bash
nvm ls
```

可以使用 `nvm use` 命令切换 `node.js` 版本。例如，切换到 `node.js 24.22.3`：

```bash
nvm use 24.22.3
```

此外，你还可以指定使用当前的 `LTS` 版本：

```bash
nvm use lts
```

## 2. 更换 nvm 的安装源

### 2.1 安装 nrm

修改全局 .npmrc 文件（对所有项目有效）：

```bash
vim ~/.npmrc

registry=https://registry.npmmirror.com/
```

### 2.2 npm 命令

通过命令行更改当前项目的源：

```bash
npm config set registry https://registry.npmmirror.com/
```

查看当前源的命令:

```bash
npm config get registry
```

### 2.3 nrm 包切换

安装依赖:

```bash
npm install -g nrm
```

查看可用的源:

```bash
nrm ls
```

```bash
$ nrm ls
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
nrm use taobao
```

测试速度（所有源的响应时间）:

根据源的响应时间长短选择最快的源使用即可。

```bash
nrm test
```

## 3. Claude Code

### 3.1 安装 Claude Code

```bash
npm install -g @anthropic-ai/claude-code
```

日常使用：

```bash
cd {$YourProjectDir}    # 切换到你的项目目录
claude --dangerously-skip-permissions -c
```

### 3.2 cc-siwtch-cli

CC-Swicth 有一个 CLI 的版本：

```bash
npm install -g cc-switch-cli
```

第一次启动 cc-switch-cli，它会自动关闭 Claude Code 的官方登录，并且新增一个模型配置。

```bash
cc-switch-cli
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

编辑 claude code 的配置文件：

```bash
vim ~/.claude.json
```

可以看到，关闭 claude 官方登陆的设置项和其他：

```json
  "hasCompletedOnboarding": true,
```
