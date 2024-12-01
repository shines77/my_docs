# Windows下编译FFmpeg（二）-- MinGW32 和 msys 1.0

## 导航

- [Windows下编译FFmpeg（一）-- MSYS2 和 mingw-w64](./Windows下编译FFmpeg（一）--MSYS2和mingw-w64.md)
- [Windows下编译FFmpeg（二）-- MinGW32 和 msys 1.0](./Windows下编译FFmpeg（二）--MinGW32和msys-1.0.md)
- [Windows下编译FFmpeg（三）-- 依赖库的下载与安装](./Windows下编译FFmpeg（三）--依赖库的下载与安装.md)
- [Windows下编译FFmpeg（四）-- FFmpeg 编译选项](./Windows下编译FFmpeg（四）--FFmpeg编译选项.md)

## 1. 前言

Windows下编译 FFmpeg 有很多种方法，一种是 MinGW + msys + gcc 环境，另一种是在 VC20XX 的命令行环境下使用 MinGW + msys + msvc 编译，

还有一种是在 cygwin 环境下编译，当然还有各种交叉编译以及 WinRT、WP8 环境下编译，这里不讨论。

但由于使用 MinGW + gcc 的方式，有时候将编出来的 FFmpeg 静态库文件给 vs 工程链接使用时，可能会出现编译错误，所以这里更推荐使用第二种方式，即 MinGW + msys + msvc 编译，这种方式不依赖 MinGW 的库。

本文是参考网上的文章综合而成，同时做为编译 FFmpeg 的笔记，如有疏漏谬误之处，望指正。

## 2. 准备工作

- **安装 Git**: 不是必须的，用于从源代码仓库克隆 FFmpeg 的源码。
- **安装 MSYS 和 MinGW**: 有两种方式，64位：MSYS 2.0 + mingw-w64，32位：MinGW + msys 1.0。
- **安装 NASM**: 一个汇编器，用于编译某些FFmpeg的汇编代码。
- **安装依赖库**: 如 libx264、libx265、libvpx 等，这些库提供了对特定视频编码的支持。

对于 MSYS 和 MinGW，为什么是这样的组合，因为对于 x86_amd64 模式：是安装 MSYS 2.0 后，其中自带了 mingw-w64；而对于 x86_32 模式：是 MinGW32 中自带了 msys 1.0，所以这是两者的包含关系。

有些时候，由于某些原因，你可能不方便或不能使用 mingw-w64，那么可以使用本文介绍的 `MinGW + msys 1.0`，注意，这个版本的 `MinGW` 是 `mingw32`，也就是 32 位版本。

## 3. MinGW 和 MSYS

由于在 Windows 下编译 FFmpeg 怎么都不可能绕过 MinGW 和 msys，所以这是必需的步骤。

### 3.1 下载 MinGW

MinGW (Minimalist GNU on Windows)：一个可自由使用和自由发布的 Windows 特定头文件和使用 GNU 工具集导入库的集合，允许你生成本地的 Windows 程序而不需要第三方 C 运行时库。

到：[https://sourceforge.net/projects/mingw/files/](https://sourceforge.net/projects/mingw/files/) 去下载即可。

1. 有一个绿色的按钮 `Download Latest Version`，点击该按钮“[Download Latest Version mingw-get-setup.exe (86.5 kB)](https://sourceforge.net/projects/mingw/files/latest/download)”会自动开始下载；

2. 如果跳转页面后没有自动下载，可点击“`Problems Downloading?`”按钮，手动点击 `direct link` 链接下载，或者选择别的镜像地址；

提示：msys 此处就不用再下载了，最新版的 mingw-get-setup.exe 中已经包含了 msys 1.0，后面安装的时候就可以看到该选项。

### 2.2 安装 MinGW (内含 msys 1.0)

点击刚下载的 `mingw-get-setup.exe` 开始安装，这是在线安装模式，点击 `Install` -> `Continue` 按钮后，进入安装 MinGW 配置器的界面，耐心等待安装完成（显示 100%）即可。

如果下载很慢或失败，可以用魔法上网。

配置信息下载完了，会进入“MinGW Installation Manager”，只勾选 `Basic Setup` 下面的以下 `Package`：

- mingw32-base: 包含了 MinGW 的工具、运行库、Windows API支持、mingw32-make、调试器，以及 GCC 的 C 编译器。

- mingw-develop-toolkit: 一个包含 MSYS 1.0 的开发工具包，会同时勾选 msys-base。

- msys-base: 一个基本的 MSYS 安装。

- mingw32-gcc-g++: mingw32-base 中包含了 gcc 的 c 编译器 mingw32-gcc，这个是 gcc 的 C++ 编译器。

其他的可先不安装，需要的时候可以随时再运行该程序来添加。

点击菜单 "Installation" -> "Apply Changes"，弹出的界面点击 `Apply` 按钮开始下载安装。

安装完成会再次要你确认，可以再安装一个 mingw32-pthread-w32，在 `MinGW Libraries` 下可以找到，勾选并安装即可。

### 2.3 环境变量

例如，你的安装路径是 `C:\MinGW`，则把 `C:\MinGW\bin` 和 `C:\MinGW\msys\1.0\bin` 目录添加到系统的 `Path` 路径即可。

打开命令行，输入 `gcc -v` 可以查看 gcc 的版本信息：

```bash
$ gcc -v

Using built-in specs.
COLLECT_GCC=C:\MinGW\bin\gcc.exe
COLLECT_LTO_WRAPPER=c:/mingw/bin/../libexec/gcc/mingw32/6.3.0/lto-wrapper.exe
Target: mingw32
Configured with: ../src/gcc-6.3.0/configure --build=x86_64-pc-linux-gnu \
--host=mingw32 --target=mingw32 --with-gmp=/mingw --with-mpfr --with-mpc=/mingw \
--with-isl=/mingw --prefix=/mingw --disable-win32-registry --with-arch=i586 \
--with-tune=generic --enable-languages=c,c++,objc,obj-c++,fortran,ada \
--with-pkgversion='MinGW.org GCC-6.3.0-1' --enable-static --enable-shared \
--enable-threads --with-dwarf2 --disable-sjlj-exceptions \
--enable-version-specific-runtime-libs --with-libiconv-prefix=/mingw \
--with-libintl-prefix=/mingw --enable-libstdcxx-debug --enable-libgomp --disable-libvtv --enable-nls
Thread model: win32
gcc version 6.3.0 (MinGW.org GCC-6.3.0-1)
```

可以看到官方最新版的 MinGW 的 gcc 版本也只更新到了 6.3.0 版本。

## 3. 参考文章

- [MinGW下载安装教程 傻瓜式操作【超详细】](https://blog.csdn.net/qq_38196449/article/details/136125995)

- [Windows下使用MinGW+msys编译ffmpeg](https://www.cnblogs.com/shines77/p/3500337.html)
