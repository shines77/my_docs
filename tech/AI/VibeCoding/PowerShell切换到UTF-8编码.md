# PowerShell 设置为 UTF-8 编码

## 前言

在 PowerShell 中切换到 UTF-8 编码，最常用且临时生效的方法是运行 `chcp 65001` 命令。若要永久生效，建议在 PowerShell 配置文件（$PROFILE）中添加该命令，或者通过系统区域设置将 Beta 版 Unicode 支持开启。

### 方法一

临时切换（当前窗口有效）

在 PowerShell 中直接输入以下命令并回车：

```powershell
chcp 65001
```

注：此方法仅在当前会话中有效，关闭窗口后失效。

### 方法二

永久切换（推荐）

通过修改 PowerShell 配置文件，让每次启动自动切换编码：

1. 打开配置文件：在 PowerShell 中输入 `notepad $PROFILE` 。如果提示找不到文件，先运行 `New-Item -ItemType File -Path $PROFILE -Force` 创建它。

2. 添加内容：在打开的记事本中添加以下一行：

```powershell
$OutputEncoding = [System.Text.Encoding]::UTF8
chcp 65001 > $null
```

3. 保存并关闭。

4. 重启 PowerShell

### 方法三

系统层面更改（彻底解决）

- 打开 Windows 控制面板 -> 区域。
- 点击 管理 选项卡。
- 点击 更改系统区域设置。
- 勾选 “Beta 版: 使用 Unicode UTF-8 提供全球语言支持”。
- 重启计算机。 

## 注意事项

- 如果执行 `chcp 65001` 后仍然显示乱码，尝试在终端窗口标题栏点击右键，选择“属性”将字体修改为支持 UTF-8 的字体（如 Consolas 或 Lucida Console）。
- 在旧版 PowerShell 5.x 中，某些 UTF-8 文件仍可能以 GBK 读取。
