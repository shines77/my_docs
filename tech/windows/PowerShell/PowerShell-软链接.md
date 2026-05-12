# Win 10 PowerShell 软链接

## 用法

在 Windows 10 的 PowerShell 中，创建目录软链接（符号链接）推荐使用 `New-Item` 命令。

**基本语法：**

```powershell
New-Item -Path "链接路径" -ItemType SymbolicLink -Target "目标路径"
```

**示例：**

```powershell
# 为文件夹 "D:\RealFolder" 在桌面创建名为 "LinkFolder" 的软链接
New-Item -Path "$env:USERPROFILE\Desktop\LinkFolder" -ItemType SymbolicLink -Target "D:\RealFolder"
```

```powershell
# 为文件夹 "C:\Users\shines77\.android" 在桌面创建名为 "avd" 的软链接，链接到 "D:\shines77\.android\avd"
New-Item -Path "$env:USERPROFILE\.android\avd" -ItemType SymbolicLink -Target "D:\shines77\.android\avd"
```

**注意：**

1. **需要管理员权限**：默认创建符号链接需以管理员身份运行 PowerShell。

   - 右键“开始” → “Windows PowerShell (管理员)”

2. **无需管理员的方法**（开启开发人员模式）：

   - 设置 → 更新和安全 → 开发者选项 → 开启“开发人员模式”。之后普通终端也可创建。

3. **区分软链接与目录联接**：

   - `SymbolicLink`：跨卷、支持相对路径，删除链接不影响目标。
   - `Junction`：仅适用于目录，且必须在同一台电脑（同一卷或不同卷均可）。用法同上，`-ItemType Junction`。

4. **删除链接**：直接删除链接文件即可，不影响目标数据。

   ```powershell
   Remove-Item "链接路径"
   ```

如果需要更强大的链接管理（如硬链接），也可使用 `cmd` 的 `mklink`，但在 PowerShell 中直接用 `New-Item` 最方便。
