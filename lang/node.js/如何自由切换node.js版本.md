# 如何自由切换 node.js 版本

本文将介绍自由切换 Node.js 版本的方法。

## 1. 使用 NVM

`NPM`（Node Version Manager）是最常用的 `node.js` 版本管理工具。它允许我们轻松安装、卸载和切换不同的 `node.js` 版本，并且支持跨平台使用。以下是 `NVM` 的安装和使用步骤:

### 1.1 安装 NVM

* 在 Linux 或 MacOS 上

可以通过以下命令安装 NVM：

    ```bash
    curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.3/install.sh | bash
    ```

安装完成后, 用下列命令检查安装是否成功, 不成功可以调整一下环境变量的顺序:

    ```bash
    nvm -v
    ```

* 在 Windows 上

对于 `Windows` 用户，可以使用 `nvm-windows`，通过以下链接下载和安装：

[Releases · coreybutler/nvm-windows · GitHub](https://github.com/coreybutler/nvm-windows/releases)

### 1.2 安装 node.js

安装 NVM 后，就可以安装多个版本的 Node.js。例如，安装最新的 LTS 版本：

    ```bash
    # 在 Linux 或 MacOS 上
    nvm install --lts

    # 在 Windows 上使用下列命令
    nvm install lts
    ```

如果你需要特定版本，比如 14.17.0，你可以通过以下命令安装：

    ```bash
    nvm install 14.17.0
    ```

### 1.3 切换 node.js 版本

查看已经安装的版本:

    ```bash
    nvm ls
    ```

可以使用 `nvm use` 命令切换 `node.js` 版本。例如，切换到 `node.js 14.17.0`：

    ```bash
    nvm use 14.17.0
    ```

此外，你还可以指定使用当前的 `LTS` 版本：

    ```bash
    nvm use lts
    ```

## 2. 更换 NVM 的源

默认的 `node.js` 下载源是: `https://nodejs.org/dist/`, 不用科学上网, 可能无法下载, 更换为 `淘宝的镜像` 即可.

    ```bash
    nvm node_mirror https://npmmirror.com/mirrors/node/
    nvm npm_mirror https://npmmirror.com/mirrors/npm/1
    ```

也可以直接修改 `NVM`的安装路径下的 `Settings.txt` 文件:

    ```bash
    root: C:\Users\YourName\AppData\Roaming\nvm
    arch: 64
    proxy: none
    originalpath: .
    originalversion:
    node_mirror: https://npmmirror.com/mirrors/node/
    npm_mirror: https://npmmirror.com/mirrors/npm/1
    ```

## 3. 参考文章

1. `[如何自由切换 Node.js 版本？]`

    [https://blog.csdn.net/W256936/article/details/142104455](https://blog.csdn.net/W256936/article/details/142104455)
