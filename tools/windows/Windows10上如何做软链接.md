# Windows 10 上如何做软链接

在 Windows 10 中，主要有 **CMD 命令**、**PowerShell** 和 **第三方 GUI 工具** 三种方式创建软链接（符号链接 / Symbolic Link）。

## 方法 1：使用 CMD（最常用）

需以 **管理员身份** 打开命令提示符（按下 `Win + X`，选择“命令提示符(管理员)”或“Windows Terminal/PowerShell”切至 CMD）。

* **创建文件软链接：**

```cmd
mklink "链接路径" "目标文件路径"
```

*示例：*

```cmd
mklink "C:\Users\Name\Desktop\notes.txt" "D:\Docs\real_notes.txt"
```

* **创建文件夹软链接（加 `/D` 参数）：**

```cmd
mklink /D "链接路径" "目标文件夹路径"
```

*示例：*

```cmd
mklink /D "C:\Users\Name\Desktop\DataFolder" "D:\Data\RealFolder"
```

## 方法 2：使用 PowerShell

同样需要以 **管理员身份** 打开 PowerShell。

* **创建文件或文件夹软链接：**

```powershell
New-Item -ItemType SymbolicLink -Path "链接路径" -Target "目标路径"
```

*示例：*

```powershell
New-Item -ItemType SymbolicLink -Path "C:\LinkFolder" -Target "D:\TargetFolder"
```

## 方法 3：使用 GUI 右键菜单（Link Shell Extension）

如果不想每次都敲命令，可以安装开源工具 **Link Shell Extension**：

1. 下载并安装 [Link Shell Extension](https://www.google.com/search?q=https://schinagl.priv.at/nt/hardlinkshellext/linkshellext.html)。
2. 右键按住目标文件或文件夹，拖动到目标位置松开，选择 **显示更多选项** -> **选取链接源 (Pick Link Source)**。
3. 在要放置软链接的位置空白处右键，选择 **创建为... (Drop As...)** -> **符号链接 (Symbolic Link)**。

## 关键注意事项

1. **免管理员权限（开发者模式）：**

默认情况下创建软链接必须拥有管理员权限。若想以普通权限创建，可以前往 **系统设置 -> 更新和安全 -> 开发者选项**，开启 **开发者模式 (Developer Mode)**。

2. **删除软链接：**

直接删除软链接文件或快捷图标即可，**不会影响源文件/源目录的内容**。

* *注意：* 对文件夹软链接进行删除时，请直接在父级目录右键删除该软链接文件夹（或使用命令 `rmdir "链接路径"`），切勿进入软链接目录内部去清空里面的文件。

3. **软链接与目录联接（Junction）的区别：**

* **软链接 (`/D`)**：支持文件和文件夹，支持跨网络共享路径。
* **目录联接 (`/J`)**：仅支持本地文件夹，兼容性更好（部分老旧软件不支持软链接但支持 Junction）。命令为 `mklink /J "链接路径" "目标路径"`。
