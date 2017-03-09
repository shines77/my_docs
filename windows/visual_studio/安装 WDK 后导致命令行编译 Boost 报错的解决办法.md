
安装 WDK 后导致命令行编译 Boost 报错的解决办法
--------------------------------------------------

# 1. 编译错误

在更新 `Windows SDK`，并且安装 `Windows Driver Kits` 以后（因为两者最好使用同一个版本，所以这个流程几乎是固定的），用命令行的方式编译 `Boost`，可能会看到如下的错误提示：

```shell
C:\Program32\Microsoft Visual Studio 14.0\VC\INCLUDE\crtdefs.h(10): fatal error C1083: 无法打开包括文件: “corecrt.h”
```

这会造成某些 `Boost` 不能正确编译，部分库没问题。其实不仅仅是 `Boost`，任何使用 `Visual Studio` 命令行方式来编译的工程，可能都会遇到此类问题。

# 2. 原因分析

我们打开 `C:\Program32\Microsoft Visual Studio 14.0\VC\INCLUDE\crtdefs.h` 文件，看到其中第 `10` 行的确包含了 `corecrt.h` 文件，如下：

```cpp
//
// crtdefs.h
//
//      Copyright (c) Microsoft Corporation. All rights reserved.
//
// Declarations used across the Visual C++ Libraries.  The lack of #pragma once
// is deliberate.
//
#include <vcruntime.h>
#include <corecrt.h>
```

而 `corecrt.h` 文件是放在 “`Windows SDK`” 的 “`UniversalCRT`” 里的 `include` 目录的，路径是：`C:\Program Files (x86)\Windows Kits\10\Include\10.0.14393.0\ucrt\corecrt.h`。

由于安装了 `WDK` 以后，导致 `C:\Program Files (x86)\Windows Kits\10\Include` 目录下除了 `10.0.14393.0` 以后，还多了一个 `wdf` 文件夹。

而当你打开 “`Visual Studio 201x`” 的命令行，读取命令行环境变量的脚本是 “`C:\Program32\Microsoft Visual Studio 14.0\Common7\Tools\VcVarsQueryRegistry.bat`”，它是通过读取 `C:\Program Files (x86)\Windows Kits\10\Include\` 路径下的所有文件夹，并且把读取到的最后一个文件夹名称作为 `WindowsSDKVersion` 和 `UniversalCRTVersion` 的值。如我前面所说的，它会认为该值为 “`wdf`”，而不是正确的值 “`10.0.14393.0`”，所以导致以 `VS` 命令行编译的时候，找不到正确的头文件，而导致编译错误。

如果你仅仅只是安装了新版本的 `Windows SDK`，则会多出一个新的文件夹，即新旧 `Windows SDK` 的版本号，例如：“`10.0.10153.0`” 和 “`10.0.14393.0`”，如按文件夹的先后顺序，一般读取到的是最新的那个版本号，即 “`10.0.14393.0`”，所以不会出错。即使读取到的是错误的旧的版本号，由于新旧版本的 `Windows SDK` 的头文件基本上是可以混用的，所以一般不会出现上面的错误。

# 3. 解决办法

知道了原因以后，那么我们来研究一下解决的办法。

`Visual Studio 201x` 的命令行统一入口是，`C:\Program32\Microsoft Visual Studio 14.0\VC\vcvarsall.bat` 文件，所有的模式，都是通过这个批处理文件出入不同的命令行参数来实现的，所有我们从这个文件开始。

查看该文件以后，它在接受命令行的启动模式参数以后，例如：`x86`、`amd64`、`x86_amd64`、`amd64_x86` 等，调用了不同的启动脚本，例如：`x86` 模式调用的是 `.\bin\vcvars32.bat` 脚本，`amd64` 模式调用的是 `.\bin\amd64\vcvars32.bat` 脚本。

先以 `.\bin\vcvars32.bat` 脚本为例，发现它对 “`Windows SDK include/lib path`” 和 “`UniversalCRT include/lib path`” 的处理是完整的，只是版本号的路径弄错了而已。该文件一开始是调用了 “`%VS140COMNTOOLS%VCVarsQueryRegistry.bat`” 文件来读取和初始化环境变量的值，所以搞定它就可以了。

“`%VS140COMNTOOLS%VCVarsQueryRegistry.bat`” 文件的实际路径是：

“`C:\Program32\Microsoft Visual Studio 14.0\Common7\Tools\VCVarsQueryRegistry.bat`”。

查看该文件，该文件有几处需要关心的，例如：

第 `50` 行：

```bat
@REM Get windows 10 sdk version number
@if not "%WindowsSdkDir%"=="" @FOR /F "delims=" %%i IN ('dir "%WindowsSdkDir%include\" /b /ad-h /on') DO @set WindowsSDKVersion=%%i\
@if not "%WindowsSDKVersion%"=="" @SET WindowsSDKLibVersion=%WindowsSDKVersion%
@if not "%WindowsSdkDir%"=="" @set WindowsLibPath=%WindowsSdkDir%UnionMetadata;...后面无关的内容我们不关心...
```

其实还有几处，可以在该文件里搜索 “`UCRTVersion`” 字符就能看到，跟上面的是类似的，就不一一列举了（下面会说）。

在 `VS` 的命令行里输入 " `dir "%WindowsSdkDir%include\" /b /ad-h /on` "，可以看到它会列出 `C:\Program Files (x86)\Windows Kits\10\Include\` 下面的所有文件夹，如下所示：

```shell
C:\> dir "%WindowsSdkDir%include\" /b /ad-h /on
10.0.14393.0
wdf
```

然后它每列出一个，就会把 `WindowsSDKVersion` 的值设置为该读取到的文件夹名称 ，这就是为什么它最终只会设置为最后一个文件夹的原因，问题就出在这里。

我尝试在注册表里搜索 “`10.0.14393.0`” 来查找一个比较稳定可靠的 `WindowsSDKVersion` 版本号的来源，可惜的是，好像注册表里并没有一个很正式的存放 `WindowsSDKVersion` 的地方，所以可以把版本号直接写到脚本里即可。虽然不够高明，但是简单暴力，如果版本号更改以后，只需要改一处即可。虽然，我们可以通过脚本来智能判断，如果不是 `x.x` 或 `x.x.x.x` 数字格式的则略过，并选择版本号最大的作为 `Windows SDK` 的最终版本号，但是较麻烦，交给读者你来完成了。

并且好像以前 `Windows SDK` 和 `Windows Driver Kits` 的版本号不是合并在一起的，使用相互独立的版本号，有些情况版本号的值大也不一定是最新的 `SDK`，虽然很少见。

我们在 “`VCVarsQueryRegistry.bat`” 文件一开始就定义了正确的版本号，如下：

```bat

@SET WindowsSDKVersionCurrent=10.0.14393.0

@call :GetWindowsSdkDir
@call :GetWindowsSdkExecutablePath32
@call :GetWindowsSdkExecutablePath64
```

注意：这里的版本号因为是路径，不能加引号，这是脚本处理的特点，跟其他编程语言不一样。

然后搜索 “`dir "%WindowsSdkDir%include\" /b /ad-h /on`"，或者找到原来的第 `51` 行（插入上面内容后的第 `53` 行）后面，插入下面这句：

```bat
@if not "%WindowsSDKVersion%"=="" @SET WindowsSDKVersion=%WindowsSDKVersionCurrent%\
```

最终修改为：

```bat
@REM Get windows 10 sdk version number
@if not "%WindowsSdkDir%"=="" @FOR /F "delims=" %%i IN ('dir "%WindowsSdkDir%include\" /b /ad-h /on') DO @set WindowsSDKVersion=%%i\
@ 下面这句是我们新加的, 强制把 WindowsSDKVersion 的值设为我们定义的版本号。
@if not "%WindowsSDKVersion%"=="" @SET WindowsSDKVersion=%WindowsSDKVersionCurrent%\
@if not "%WindowsSDKVersion%"=="" @SET WindowsSDKLibVersion=%WindowsSDKVersion%
@if not "%WindowsSdkDir%"=="" @set WindowsLibPath=%WindowsSdkDir%UnionMetadata;...后面无关的内容我们不关心...
```

你可以继续搜索 “`dir "%WindowsSdkDir%include\" /b /ad-h /on`"，找到下一个要修改的地方，在搜索的地方的下一句插入上面那条语句。除了前面所说的第 `53` 行以外，另一处大概在第 `77` 行附近，共两个地方。

同样的，搜索 “`dir "%UniversalCRTSdkDir%include\" /b /ad-h /on`”，同样，在下一句插入如下的语句：

```bat
@if not "%UCRTVersion%"=="" @SET UCRTVersion=%WindowsSDKVersionCurrent%
```

效果如下所示：

```bat
@if "%UniversalCRTSdkDir%"=="" exit /B 1
@FOR /F "delims=" %%i IN ('dir "%UniversalCRTSdkDir%include\" /b /ad-h /on') DO @SET UCRTVersion=%%i
@if not "%UCRTVersion%"=="" @SET UCRTVersion=%WindowsSDKVersionCurrent%
@exit /B 0
```

类似前面，也有两个地方，大概分布在第 `295` 和 `305` 行。

这里要注意一个地方，`%WindowsSDKVersion%` 后面是要带 “`\`” （反斜杠）的，而 `%UCRTVersion%` 的后面是不需要带 “`\`” 的，有一点小差别。

除了文件一开始添加的版本号定义以外，一共修改了四个地方，插入四条语句（两种不同的语句，每种给插入两条）。

# 4. 测试

好了，现在重新打开你的 “`Visual Studio 201x`” 命令行，输入如下命令：

```shell
C:\> echo %WindowsSDKVersion%
10.0.14393.0\

C:\> echo %UCRTVersion%
10.0.14393.0
```

如果看到的是你设置的 `版本号`，而且 `%WindowsSDKVersion%` 后多一个 “`\`” 字符，则说明 `OK` 了，应该可以正常编译了。

`Enjoy it!`

你还可以检查如下一些路径是否正确：

```shell
C:\> echo %WindowsSdkDir%
C:\Program Files (x86)\Windows Kits\10\

C:\> echo %UniversalCRTSdkDir%
C:\Program Files (x86)\Windows Kits\10\
```

`搞定！`
