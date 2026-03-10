# Zig 的安装与使用

## 1. 安装 Zig

### 1.1 使用 Scoop 安装

推荐使用 Scoop 工具进行安装。Scoop 的 main 仓库提供最新的 release 版本，而 versions 仓库提供 nightly 版本。

#### 1.1.1 安装 Scoop

必须在非管理员权限的 PowerShell 下执行以下命令：

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
Invoke-RestMethod -Uri https://get.scoop.sh | Invoke-Expression
```

引用：[Scoop官网](https://scoop.sh/#/)

如果你已经安装过 Scoop，可以使用下列命令升级到最新版：

```powershell
scoop update
```

#### 1.1.2 安装 Zig

Scoop 的 main 仓库提供最新的 release 版本。

```powershell
scoop bucket add main
scoop install main/zig
```

Scoop 的 versions 仓库提供最新的 nightly 版本（开发版）。

```powershell
scoop bucket add versions
scoop install versions/zig-dev
```

**提示**

- 使用 scoop reset zig-dev 或 scoop reset zig 可以在 nightly 和 release 版本之间切换。
- 使用 scoop install zig@0.11.0 可以安装特定版本，同理，scoop reset zig@0.11.0 可以切换到该指定版本。

### 1.2 其他的包管理器

也可以使用诸如 [WinGet](https://github.com/microsoft/winget-cli) 或 [Chocolatey](https://chocolatey.org/) 等包管理器。

[WinGet](https://github.com/microsoft/winget-cli)：

```bash
winget install -e --id zig.zig
```

[Chocolatey](https://chocolatey.org/)：

```bash
choco install zig
```

### 1.3 手动安装

从官方 [发布页面](https://ziglang.org/zh/download/) 下载对应的 Zig 版本，大多数用户应选择 `zig-windows-x86_64`。

解压后，将包含 `zig.exe` 的目录路径添加到系统的 Path 环境变量中。

你可以自己手动添加（推荐），也可以通过以下 PowerShell 命令来完成：

`System` （全局环境变量）：

```powershell
[Environment]::SetEnvironmentVariable("Path", [Environment]::GetEnvironmentVariable("Path", "Machine") + ";C:\your-path\zig-windows-x86_64-your-version", "Machine")
```

`User` （当前用户环境变量）：

```powershell
[Environment]::SetEnvironmentVariable("Path", [Environment]::GetEnvironmentVariable("Path", "User") + ";C:\your-path\zig-windows-x86_64-your-version", "User")
```

**提示**

- System 对应系统全局环境变量，User 对应当前用户环境变量。如果是个人电脑，两者通常没有太大区别。

- 请确保将 `C:\your-path\zig-windows-x86_64-your-version` 替换为你的实际解压路径。路径前的分号 `;` 是必需的，并非拼写错误，它用于在 Path 变量中分隔多个路径。


## X. 参考文章

- [Zip语言](https://course.ziglang.cc/environment/install-environment)
