# Windows 下编译 FFmpeg

## 1. 前言

Windows下编译 FFmpeg 有很多种方法，一种是 MinGW + msys + gcc 环境，一种是在 VC20XX 的命令行环境下使用 MinGW + msys + msvc 编译，

还有一种是在 cygwin 环境下编译，当然还有各种交叉编译以及 WinRT、WP8 环境下编译，这里不讨论。

但由于使用 MinGW + gcc 的方式，有时候将编出来的 FFmpeg 静态库文件给 vs 工程链接使用时，可能会出现编译错误，所以这里使用更稳妥的第二种方式，MinGW + msys + msvc 编译，不依赖 MinGW 的库。

本文是参考网上的文章综合而成，同时做为编译 FFmpeg 的笔记，如有疏漏谬误之处，望指正。

## 2. 准备工作

您最好新建一个目录专门用来保存以下下载的文件。例如：F:\ffmpeg 。这里更推荐安装 mingw-w64，所以 2.1 ~ 2.3 小节可以跳过。

### 2.1 下载 MinGW 和 MSYS

由于在 Windows 下编译 FFmpeg 怎么都不可能绕过 MinGW 和 msys，所以这是必需的步骤。

MinGW (Minimalist GNU on Windows)：一个可自由使用和自由发布的 Windows 特定头文件和使用 GNU 工具集导入库的集合，允许你生成本地的 Windows 程序而不需要第三方 C 运行时库。

到：[https://sourceforge.net/projects/mingw/files/](https://sourceforge.net/projects/mingw/files/) 去下载即可。

1. 有一个绿色的按钮 `Download Latest Version`，点击该按钮“[Download Latest Version mingw-get-setup.exe (86.5 kB)](https://sourceforge.net/projects/mingw/files/latest/download)”会自动开始下载；

2. 如果跳转页面后没有自动下载，可点击“`Problems Downloading?`”按钮，手动点击 `direct link` 链接下载，或者选择别的镜像地址；

提示：msys 此处就不用再下载了，最新版的 mingw-get-setup.exe 中已经包含了 msys1.0，后面安装的时候就可以看到该选项。

### 2.2 安装 MinGW

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
Configured with: ../src/gcc-6.3.0/configure --build=x86_64-pc-linux-gnu --host=mingw32 --target=mingw32 --with-gmp=/mingw --with-mpfr --with-mpc=/mingw --with-isl=/mingw --prefix=/mingw --disable-win32-registry --with-arch=i586 --with-tune=generic --enable-languages=c,c++,objc,obj-c++,fortran,ada --with-pkgversion='MinGW.org GCC-6.3.0-1' --enable-static --enable-shared --enable-threads --with-dwarf2 --disable-sjlj-exceptions --enable-version-specific-runtime-libs --with-libiconv-prefix=/mingw --with-libintl-prefix=/mingw --enable-libstdcxx-debug --enable-libgomp --disable-libvtv --enable-nls
Thread model: win32
gcc version 6.3.0 (MinGW.org GCC-6.3.0-1)
```

可以看到官方最新版的 MinGW 的 gcc 版本也只到了 6.3.0 版本，如果想用更新的 gcc 版本，可以安装 mingw-w64，下一节会介绍。

### 2.4 安装 mingw-w64

首先，进去官网地址：[https://www.mingw-w64.org](https://www.mingw-w64.org)，左侧点击 “Downloads”，然后在 Downloads 页面是右侧，找到 “MinGW-W64-builds” 并点击，会跳转到如下内容：

**MinGW-W64-builds**

Installation: [GitHub](https://github.com/niXman/mingw-builds-binaries/releases)

**MSYS2**

Installation: [GitHub](http://msys2.github.io/)

点击上面第一个 GitHub 的链接：[https://github.com/niXman/mingw-builds-binaries/releases](https://github.com/niXman/mingw-builds-binaries/releases)

这里选择 `x86_64-14.2.0-release-posix-seh-msvcrt-rt_v12-rev0.7z` 版本。

我们要开发兼容 Linux、Unix、Mac OS 操作系统下的程序，所以要选择 `posix` 版本，`seh` 是异常处理模式，不支持 32 位异常处理。选择 msvcrt 链接方式是为了更好的兼容性，MSYS2 官方比较推荐新的 ucrt 链接方式。

**posix 和 win32 的区别**

POSIX 是一种 UNIX API 标准，而 Win32 是 Windows 的API标准。这两者之间有一些区别，例如在 mingw-w64 中，使用 posix 线程将启用 C++11/C11 多线程功能，并使 libgcc 依赖于 libwinpthreads。而使用 win32 线程则不会启用 C++11 多线程功能。

**dwarf 和 seh 的区别**

DWARF（DW2，dwarf-2）和 SEH（零开销 exception）是两种不同的异常处理模型。DWARF 仅适用于 32 位系统，没有永久的运行时开销，但需要整个调用堆栈被启用。SEH 将可用于 64 位 GCC 4.8。

**msvcrt 和 ucrt 的区别**

MSVCRT（Microsoft Visual C++ Runtime）和 UCRT（Universal C Runtime）是 Microsoft Windows 上的两种C标准库变体。MSVCRT 在所有 Microsoft Windows 版本中都默认可用，但由于向后兼容性问题，它已经过时，不兼容 C99 并且缺少一些功能。而 UCRT 是一个较新的版本，也是 Microsoft Visual Studio 默认使用的版本。它应该像使用 MSVC 编译的代码一样工作和表现。

关于 `msvcrt` 和 `ucrt` 两种链接方式的选择可以参考以下问答：

[在 msys2 中的 mingw64 、ucrt64 、clang64 的区别与相同点有啥？](https://www.zhihu.com/question/463666011)

由于以上 github 仓库提供的 `mingw-w64 Release` 的版本选择不是很多，更多的 mingw-w64 版本可以在这里找到：[windows上安装mingw教程及mingw64国内下载地址汇总](https://blog.csdn.net/FL1623863129/article/details/142673029) 。

下载完成后，把压缩包里的目录和文件解压到 C 盘，并把 mingw-w64 的根目录更名为 'C:\mingw64'，其实它本来就叫这个名字，不用改。

**添加环境变量**

类似的，把 `C:\mingw64\bin` 目录添加到系统的 `Path` 路径即可。

另外，这种安装方式是不带 `msys` 的，你还需要单独安装一个 `msys2`，链接见上，并配置 `msys2` 的环境变量。

**其他**

这个版本跟前面介绍的 `MinGW` 32 位版本是冲突的，系统环境变量里只能配置其中一个，建议选择 `ming32-w64` 。

**gcc版本**

检查 gcc 的版本，可以看到是 `14.2.0` 。

```bash
$ gcc -v

Using built-in specs.
COLLECT_GCC=C:\mingw64\bin\gcc.exe
COLLECT_LTO_WRAPPER=C:/mingw64/bin/../libexec/gcc/x86_64-w64-mingw32/14.2.0/lto-wrapper.exe
Target: x86_64-w64-mingw32
Configured with: ../../../src/gcc-14.2.0/configure --host=x86_64-w64-mingw32 --build=x86_64-w64-mingw32 \
--target=x86_64-w64-mingw32 --prefix=/mingw64 \
--with-sysroot=/c/buildroot/x86_64-1420-posix-seh-msvcrt-rt_v12-rev0/mingw64 \
--enable-host-shared --disable-multilib --enable-languages=c,c++,fortran,lto --enable-libstdcxx-time=yes \
--enable-threads=posix --enable-libgomp --enable-libatomic --enable-lto --enable-graphite --enable-checking=release \
--enable-fully-dynamic-string --enable-version-specific-runtime-libs --enable-libstdcxx-filesystem-ts=yes \
--disable-libssp --disable-libstdcxx-pch --disable-libstdcxx-debug --enable-bootstrap --disable-rpath \
--disable-win32-registry --disable-nls --disable-werror --disable-symvers --with-gnu-as --with-gnu-ld \
--with-arch=nocona --with-tune=core2 --with-libiconv --with-system-zlib \
--with-gmp=/c/buildroot/prerequisites/x86_64-w64-mingw32-static \
--with-mpfr=/c/buildroot/prerequisites/x86_64-w64-mingw32-static \
--with-mpc=/c/buildroot/prerequisites/x86_64-w64-mingw32-static \
--with-isl=/c/buildroot/prerequisites/x86_64-w64-mingw32-static \
--with-pkgversion='x86_64-posix-seh-rev0, Built by MinGW-Builds project' \
--with-bugurl=https://github.com/niXman/mingw-builds \
LD_FOR_TARGET=/c/buildroot/x86_64-1420-posix-seh-msvcrt-rt_v12-rev0/mingw64/bin/ld.exe \
--with-boot-ldflags='-pipe -fno-ident -L/c/buildroot/x86_64-1420-posix-seh-msvcrt-rt_v12-rev0/mingw64/opt/lib \
-L/c/buildroot/prerequisites/x86_64-zlib-static/lib -L/c/buildroot/prerequisites/x86_64-w64-mingw32-static/lib  \
-Wl,--disable-dynamicbase -static-libstdc++ -static-libgcc'
Thread model: posix
Supported LTO compression algorithms: zlib
gcc version 14.2.0 (x86_64-posix-seh-rev0, Built by MinGW-Builds project)
```

## x. 参考文章

- [MinGW下载安装教程 傻瓜式操作【超详细】](https://blog.csdn.net/qq_38196449/article/details/136125995)

- [windows上安装mingw教程及mingw64国内下载地址汇总](https://blog.csdn.net/FL1623863129/article/details/142673029)

- [【软件教程】MingW-W64-builds不同版本之间的区别](https://blog.csdn.net/zhangjiuding/article/details/129556458)

- [Windows下使用MinGW+msys编译ffmpeg](https://www.cnblogs.com/shines77/p/3500337.html)
