# Ubuntu 20.04 如何安装 CMake 3.20 以上的版本

## 问题

> 怎么在 Ubuntu 20.04 中安装 CMake 3.20 以上的版本，默认的 CMake 版本是 3.16？

## 回答

（来自 [Google - Gemini 2.5 Pro](https://gemini.google.com/)）

好的，在 Ubuntu 20.04 (Focal Fossa) 上，默认的 apt 仓库提供的 CMake 版本确实是 3.16.x。要安装 CMake 3.20 或更高版本，你有以下几种常用方法：

## **方法一：使用 Kitware APT 仓库 (推荐)**

Kitware 是 CMake 的开发者，他们提供了一个官方的 APT 仓库，其中包含最新版本的 CMake。这是最推荐的方法，因为它能与系统的包管理器 (apt) 很好地集成，方便后续更新。

1. 卸载旧版本 (可选但推荐):

如果系统中已经通过 apt 安装了旧版本的 CMake，最好先卸载它以避免冲突。

```bash
sudo apt update
sudo apt remove cmake
sudo apt autoremove
```

2. **获取 Kitware APT 仓库的 GPG 密钥并添加**:

```bash
wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | sudo tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null
```

3. 添加 Kitware APT 仓库源:

```bash
bash echo 'deb \[signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg\] https://apt.kitware.com/ubuntu/ focal main' | sudo tee /etc/apt/sources.list.d/kitware.list >/dev/null
```

\* 注意: 上面的命令中 focal 对应 Ubuntu 20.04 的代号。如果你在其他 Ubuntu 版本上操作，需要替换成相应的代号 (如 bionic for 18.04, jammy for 22.04)。

4. **更新包列表并安装 CMake**:

```bash
sudo apt update
sudo apt install cmake
```

apt 现在会从 Kitware 仓库中查找并安装最新可用的 CMake 版本。

5. **验证安装**:

```bash
cmake --version
```

输出应该显示你刚刚安装的新版本 (例如 3.20 或更高)。

## **方法二：使用 Snap 安装**

Snap 是 Canonical 开发的一种通用软件包格式，通常包含较新版本的软件。

1. **卸载旧版本**:

```bash
sudo apt remove cmake
sudo apt autoremove
```

2. 安装 Snap (如果尚未安装):

Ubuntu 20.04 通常没有预装 snapd ，可以运行：

```bash
sudo apt update
sudo apt install snapd
```

3. **通过 Snap 安装 CMake**:

```bash
sudo snap install cmake --classic
```

* \--classic 参数是必需的，因为它允许 CMake 访问系统文件，这对于构建工具通常是必要的。

4. **设置 CMake 路径**

该方法安装的 CMake 在 `/usr/local/bin` 目录下，如果不知，可使用 `which cmake` 命令查看。

我们要把 `/usr/local/bin` 添加到系统目录中，打开 `~/.bashrc` 文件：

```bash
sudo vim ~/.bashrc
```

在文件末尾添加下面这一句：

```bash
export PATH="/usr/local/bin:$PATH"
```

然后运行 `source ~/.bashrc` 使更改生效。或者直接在命令行里执行上面那条命令，但重启后会失效，最好还是添加到 `~/.bashrc` 文件中。

5. **验证安装**:

```bash
cmake --version
```

## **方法三：使用 pip 安装 (Python 环境)**

如果你主要在 Python 项目或虚拟环境中使用 CMake，可以通过 pip 安装。

1. **卸载旧版本**:

```bash
sudo apt remove cmake
sudo apt autoremove
```

2. **确保有 pip**:

```bash
sudo apt update
sudo apt install python3-pip
```

3. **使用 pip 安装 CMake**:

```bash
pip install cmake
```

或者，如果你想在用户级别安装而不是系统范围 (推荐):

```bash
pip install --user cmake
```

这通常会安装一个较新的 CMake 版本。

4. **设置 CMake 路径**

这里也要设置 CMake 的路径，步骤和方法二的一样，不知道 CMake 安装路径的话，可以使用 `which cmake` 查看，这里不再赘述。

5. **验证安装**:

```bash
cmake --version
```

或者，如果安装到用户目录，可能需要使用 `~/.local/bin/cmake --version`，或者确保 `~/.local/bin` 在系统的 PATH 中。

## **方法四：下载官方预编译二进制文件**

你可以直接从 CMake 官网下载预编译好的 Linux 二进制文件 (.sh 脚本安装包)。

1. 访问 CMake 下载页面:

   前往 <https://cmake.org/download/>

2. 下载适用于 Linux x86\_64 的 .sh 安装脚本:

   找到最新稳定版本 (或你需要的特定版本 \>= 3.20)，下载名为 cmake-\<version\>-linux-x86\_64.sh 的文件。例如，下载 cmake-3.29.0-linux-x86\_64.sh。

3. **赋予脚本执行权限**:

```bash
chmod +x cmake-*-linux-x86_64.sh
```

(将 \* 替换为你下载的具体版本号)

4. 运行安装脚本:

你可以选择安装到系统范围 (/usr/local，需要 sudo) 或用户目录。

* **安装到 /usr/local (推荐用于系统级可用)**:

    ```bash
    sudo ./cmake-*-linux-x86_64.sh --prefix=/usr/local --skip-license
    ```

  * --skip-license 会跳过交互式许可证确认。
  * 安装到 /usr/local/bin 的程序通常会自动包含在系统的 PATH 中。

* **安装到用户主目录 (无需 sudo)**:

    ```bash
    ./cmake-*-linux-x86_64.sh --prefix=$HOME/.local --skip-license
    ```

    你需要确保 $HOME/.local/bin 在你的 PATH 环境变量中。可以将以下行添加到你的 \~/.bashrc 或 \~/.zshrc 文件中：

    ```bash
    export PATH="$HOME/.local/bin:$PATH"
    ```

    然后运行 `source ~/.bashrc` (或 .zshrc) 使更改生效。

5. 验证安装:

打开一个新的终端窗口（如果修改了 .bashrc），然后运行：

```bash
cmake --version
```

## **总结**

* **最推荐**: **方法一 (Kitware APT 仓库)**，因为它与系统包管理集成最好。
* **简单方便**: **方法二 (Snap)** 或 **方法三 (pip)**，尤其是在隔离环境或 Python 项目中时。
* **直接控制**: **方法四 (官方二进制文件)**，如果你不想添加 PPA 或使用 Snap，并且需要特定版本。

选择哪种方法取决于你的具体需求和偏好。对于大多数用户来说，使用 Kitware 的 APT 仓库是最佳选择。

## **参考来源**

1\. [https://github.com/AXIHIXA/Memo](https://github.com/AXIHIXA/Memo)
