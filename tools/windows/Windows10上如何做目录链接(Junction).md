# Windows 10 上如何做目录链接(Junction)

在 Windows 10 中创建**目录联接（Directory Junction）**非常简单，且相比软链接，它最大的优势是**通常不需要管理员权限**（只要你在目标位置有写入权限）。

以下是常用的创建方式：

## 方法 1：使用 CMD（最直接）

打开普通或管理员命令提示符，使用带有 `/J` 参数的 `mklink` 命令：

```cmd
mklink /J "链接路径" "目标路径"
```

* **示例：** 将 `D:\Games\Steam` 联接到 `C:\Steam`

```cmd
mklink /J "C:\Steam" "D:\Games\Steam"
```

*(执行后，访问 `C:\Steam` 就会直接指向 `D:\Games\Steam` 的内容)*

## 方法 2：使用 PowerShell

在 PowerShell 中，直接使用原生命令 `New-Item`，并将类型指定为 `Junction`：

```powershell
New-Item -ItemType Junction -Path "链接路径" -Target "目标路径"
```

* **示例：**

```powershell
New-Item -ItemType Junction -Path "C:\Steam" -Target "D:\Games\Steam"
```

## 目录联接（Junction）的使用注意点

* **无需管理员权限**：在普通用户有权写入的目录（如 `C:\Users\YourName\...`）下创建 Junction，不需要管理员身份运行终端。
* **仅限本地文件夹**：Junction 只能作用于**文件夹**（不能是单个文件），且目标路径必须是**本地磁盘**（不支持网络共享路径 UNC）。
* **软件兼容性极佳**：很多老旧软件或游戏无法识别普通软链接，但通常能完美兼容 Junction。
* **如何安全删除：**

  * 在文件资源管理器中直接**右键删除**该联接文件夹图标即可。
  * 在 CMD 中可以使用：`rmdir "链接路径"`
  * **切记：** 只删除联接文件夹本身，千万不要进入联接文件夹内部去全选清空里面的内容，否则会把源文件夹里的真实文件一并删除。
