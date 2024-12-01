# Windows 下编译 FFmpeg

## 1. 前言

Windows下编译 FFmpeg 有很多种方法，一种是 MinGW + msys + gcc 环境，一种是在 VC20XX 的命令行环境下使用 MinGW + msys + msvc 编译，

还有一种是在 cygwin 环境下编译，当然还有各种交叉编译以及 WinRT、WP8 环境下编译，这里不讨论。

但由于使用 MinGW + gcc 的方式，有时候将编出来的 FFmpeg 静态库文件给 vs 工程链接使用时，可能会出现编译错误，所以这里使用更稳妥的第二种方式，MinGW + msys + msvc 编译，不依赖 MinGW 的库。

本文是参考网上的文章综合而成，同时做为编译 FFmpeg 的笔记，如有疏漏谬误之处，望指正。

## 2. 准备工作

您最好新建一个目录专门用来保存以下下载的文件。例如：F:\ffmpeg 。

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

### 2.4 mingw-w64 安装

首先，进去官网地址：[https://www.mingw-w64.org](https://www.mingw-w64.org)，左侧点击 “Downloads”，然后在 Downloads 页面是右侧，找到 “Sources” 并点击。

会看到如下内容，可以直接点击“SourceForge”链接跳转到 sourceforge.net 的下载目录，或者点击其中某个版本号直接下载某个版本，例如选择 `11.0.0`，则是下载 `mingw-w64-v11.0.0.zip` 文件。更推荐自己手动选择版本，最新的版本已到 `12.0.0` 。

```
Sources

Tarballs for the mingw-w64 sources are hosted on SourceForge.

The latest version from the 11.x series is 11.0.0.

The latest version from the 10.x series is 10.0.0.

The latest version from the 9.x series is 9.0.0.

The latest version from the 8.x series is 8.0.2.

The latest version from the 7.x series is 7.0.0.

The latest version from the 6.x series is 6.0.0.

The latest version from the 5.x series is 5.0.4.

The old wiki has instructions for building native and cross toolchains.

Details on how to get the mingw-w64 code from Git and an Git-web viewer are available on SourceForge.
```

下载完成后，例如我下载的是：mingw-w64-v11.0.1.zip，把其中的 `mingw-w64-v11.0.1` 解压到 `C:\`，并把 `C:\mingw-w64-v11.0.1` 目录更名为 'C:\mingw-w64' 。

**添加环境变量**

类似的，把 `C:\mingw-w64\bin` 目录添加到系统的 `Path` 路径即可。

另外，这种安装方式是不带 `msys` 的，你可能还需要单独安装一个 `msys` 。

**其他**

关于 mingw64-w64 的更多信息，可以参考：[windows上安装mingw教程及mingw64国内下载地址汇总](https://blog.csdn.net/FL1623863129/article/details/142673029)

## x. 参考文章

- [MinGW下载安装教程 傻瓜式操作【超详细】](https://blog.csdn.net/qq_38196449/article/details/136125995)

- [windows上安装mingw教程及mingw64国内下载地址汇总](https://blog.csdn.net/FL1623863129/article/details/142673029)

- [Windows下使用MinGW+msys编译ffmpeg](https://www.cnblogs.com/shines77/p/3500337.html)
