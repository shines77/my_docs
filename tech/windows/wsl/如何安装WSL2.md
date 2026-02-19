# 如何安装WSL2

## 1. 系统要求

WSL2 要求 Windows 10 的 1903 或内部版本 18362 或者更高版本。

按 `Win键` + R，然后键入“winver”，按下Enter键即可看到 Windows 的版本号。

## 2. 开启 Hyper-V 虚拟机

要使用 WSL2，必须开启 Hyper-V 虚拟机功能，有两种方式开启。

### 2.1 方法一：命令行

以管理员身份打开 PowerShell，执行下列命令。

作用是启用适用于 Linux 的 Windows 子系统：

```powershell
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
```

执行下列的命令，开启 Hyper-V 虚拟机功能：

```powershell
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
```

执行完了，要重启计算机才能生效。

### 2.2 方法二：直接添加

以上两个功能可以通过控制面板来打开，找到 “控制面板” -> “程序” -> “程序和功能” -> “启用和关闭 Windows 功能”，勾选 “适用于 Linux 的 Windows 子系统” 和 “虚拟机平台”，分别对应上面的两个命令行。

勾选两者后，确定，安装好后，需要重启后才能生效。

## 3. 下载 Linux 内核更新包

点击 [旧版页面](https://docs.microsoft.com/zh-cn/windows/wsl/wsl2-kernel)，会跳转到 [最新的页面](https://learn.microsoft.com/zh-cn/windows/wsl/install-manual#step-4---download-the-linux-kernel-update-package) 的第 4 步，前面 3 步我们已经做过了。

1. 下面是该页面推荐的两个版本，下载最新的更新包，包括 x64 和 arm64 的版本：

- [WSL2 Linux 内核更新包适用于 x64 计算机](https://wslstorestorage.blob.core.windows.net/wslblob/wsl_update_x64.msi)
- [适用于 ARM64 计算机的 WSL2 Linux 内核更新包](https://wslstorestorage.blob.core.windows.net/wslblob/wsl_update_arm64.msi)

2. 运行在上一步中下载的更新包。双击以运行 - 系统会提示你输入提升的权限，选择“是”以批准此安装。安装完成后，转到下一小节。

## 4. 将 WSL 2 设置为默认版本

在安装新的 Linux 分发版时将 WSL 2 设置为默认版本。（如果只想装 WSL 1，请跳过此步骤）。

如果你之前装过 WSL (1代)，可以先把它升级为 WSL 2，命令如下，例如：

```powershell
wsl —set-version Ubuntu-18.04 2
```

其中 Ubuntu-18.04 是你以前安装的 WSL Linux 系统的名称。你可以随时将上述命令中的“2”改成“1”来让 WSL 回滚到 WSL (1)。

同时，指定 WSL 2 为默认版本，命令如下：

```powershell
wsl --set-default-version 2
```

## 5. 安装所选 Linux 分发版

### 5.1 下载安装包

打开 [Microsoft Store](ms-windows-store://collection?CollectionId=LinuxDistros) 并选择你喜欢的 Linux 分发版。

下面是一些比较常用的 Linux 分发版本：

- Ubuntu：

  - [Ubuntu](https://apps.microsoft.com/detail/9PDXGNCFSCZV)
  - [Ubuntu 24.04.1 LTS](https://apps.microsoft.com/detail/9NZ3KLHXDJP5)
  - [Ubuntu 22.04.5 LTS](https://apps.microsoft.com/detail/9PN20MSR04DW)
  - [Ubuntu 20.04.6 LTS](https://apps.microsoft.com/detail/9MTTCL66CPXJ)
  - [Ubuntu 20.04 LTS](https://apps.microsoft.com/detail/9N6SVWS3RX71)
  - [Ubuntu 18.04.6 LTS](https://apps.microsoft.com/detail/9PNKSF5ZN4SW)
  - [Ubuntu 18.04 LTS](https://apps.microsoft.com/detail/9N9TNGVNDL3Q)
  - [Ubuntu（预览版）](https://apps.microsoft.com/detail/9P7BDVKVNXZ6)

- Debian：

  - [Debian](https://apps.microsoft.com/detail/9MSVKQC78PK6)

- Arch Linux：

  - [Arch WSL](https://apps.microsoft.com/detail/9MZNMNKSM73X)

- Fedora：

  - [Fedora WSL](https://apps.microsoft.com/detail/9NPCP8DRCHSN)

- deepin：

  - [deepin WSL](https://apps.microsoft.com/detail/9P6HT7L0QGRH)

### 5.2 安装

有两种方法可以安装。

1. 直接从 [Microsoft Store](ms-windows-store://collection?CollectionId=LinuxDistros) 的页面里直接下载安装，这里不赘述了。

2. 用命令行来安装：

先用命令行查询一下微软在线商店有哪些 Linux 分发版本，请输入：

```powershell
wsl.exe --list --online
```

结果如下：

```powershell
以下是可安装的有效分发的列表。
默认分发用 “*” 表示。
使用 'wsl --install -d <Distro>' 安装。

  NAME                            FRIENDLY NAME

* Ubuntu                          Ubuntu
  Debian                          Debian GNU/Linux
  kali-linux                      Kali Linux Rolling
  Ubuntu-20.04                    Ubuntu 20.04 LTS
  Ubuntu-22.04                    Ubuntu 22.04 LTS
  Ubuntu-24.04                    Ubuntu 24.04 LTS
  OracleLinux_7_9                 Oracle Linux 7.9
  OracleLinux_8_10                Oracle Linux 8.10
  OracleLinux_9_5                 Oracle Linux 9.5
  openSUSE-Leap-15.6              openSUSE Leap 15.6
  SUSE-Linux-Enterprise-15-SP6    SUSE Linux Enterprise 15 SP6
  openSUSE-Tumbleweed             openSUSE Tumbleweed
```

实际上自己去 [Microsoft Store](ms-windows-store://collection?CollectionId=LinuxDistros) 找能找到更多的可安装版本，但常用的这里也都有了。

在想好要安装哪个分发版本后，可以用以下命令安装：

```powershell
wsl --install -d <发行版>
```

例如：

```powershell
wsl --install -d Ubuntu-24.04
```

安装过程中会让我们填写 Linux 的登录密码。

## 6. 参考文章

- [旧版 WSL 的手动安装步骤](https://learn.microsoft.com/zh-cn/windows/wsl/install-manual)

- [如何使用 WSL 在 Windows 上安装 Linux](https://learn.microsoft.com/zh-cn/windows/wsl/install-manual)

- [windows 下完整的 linux 内核 -- WSL2 升级与体验](https://cloud.tencent.com/developer/article/2031611)

- [双系统系列：WSL2-适用于 Linux 的 Windows 子系统（安装）](https://cloud.tencent.com/developer/article/1935774?policyId=1003)

- [WSL2：Windows 亲生的 Linux 子系统](https://cloud.tencent.com/developer/article/1861285?policyId=1003)
