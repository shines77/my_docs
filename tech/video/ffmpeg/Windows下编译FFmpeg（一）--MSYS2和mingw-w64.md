# Windows 下编译 FFmpeg（一）-- MSYS2 和 mingw-w64

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

## 2. 准备环境

- **安装 Git**: 不是必须的，用于从源代码仓库克隆 FFmpeg 的源码。
- **安装 MSYS 和 MinGW**: 有两种方式，64位：MSYS 2.0 + mingw-w64，32位：MinGW + msys 1.0。
- **安装 NASM**: 一个汇编器，用于编译某些FFmpeg的汇编代码。
- **安装依赖库**: 如 libx264、libx265、libvpx 等，这些库提供了对特定视频编码的支持。

对于 MSYS 和 MinGW，为什么是这样的组合，因为对于 x86_amd64 模式：是安装 MSYS 2.0 后，其中自带了 mingw-w64；而对于 x86_32 模式：是 MinGW32 中自带了 msys 1.0，所以这是两者的包含关系，由于现在的 CPU 和 操作系统基本都已经是 64 位的，所以推荐使用 MSYS 2.0 + mingw-w64，对于 MinGW + msys 1.0 的安装和使用将在系列文章的下一篇文章中介绍。

## 3. MSYS 2.0

MSYS2 是一个工具和库的集合，为用户提供一个易于使用的环境来构建、安装和运行本机 Windows 软件。

它包括一个名为 mintty 的命令行终端、bash、版本控制系统如 git 和 subversion、工具如 tar 和 awk，甚至构建系统如 autotools，都是基于修改版的 Cygwin 实现的。尽管其中一些核心部分基于 Cygwin，但 MSYS2 的主要重点是为本机 Windows 软件提供构建环境，Cygwin 使用的部分被保持在最小限度。

为了提供易于安装软件包的方式以及保持其更新，MSYS2 提供了一个的包管理系统 —— Pacman。

MSYS2 提供了最新的 GCC、mingw-w64、CPython、CMake、Meson、OpenSSL、FFmpeg、Rust、Ruby 等本地构建工具。

### 3.1 安装 MSYS 2.0

MSYS2 官网：[https://www.msys2.org/](https://www.msys2.org/)

进入官网，"Installation" 下有下载的链接，例如：

1. Download the installer: [msys2-x86_64-20241116.exe](https://github.com/msys2/msys2-installer/releases/download/2024-11-16/msys2-x86_64-20241116.exe)

注：MSYS2 安装要求 64 位的 Windows 10 或更高版本，如果达不到这个要求可以使用下篇文章介绍的 “MinGW + msys 1.0” 的方式。

下载并安装之后，会启动一个 MSYS2 的 `UCRT64` 的环境：

```bash
user@desktop UCRT64 ~
```

### 3.2 pacman 更换源

由于 MSYS 2.0 的默认安装源可能速度很慢，所以要把它改为国内访问更快的安装源，比如：科大源或清华源。

一般的，pacman 的镜像源文件位置位于 `/etc/pacman.d/`：

```bash
$ ls /etc/pacman.d

gnupg               mirrorlist.clang64  mirrorlist.mingw32  mirrorlist.msys
mirrorlist.clang32  mirrorlist.mingw    mirrorlist.mingw64  mirrorlist.ucrt64
```

不同的镜像源文件，对应着不同的环境。

以 `mirrorlist.ucrt64` 为例，进行修改，一般选择科大源，除非访问失效，再尝试换成别的源。

```bash
$ nano /etc/pacman.d/mirrorlist.ucrt64

# See https://www.msys2.org/dev/mirrors

## Primary
# Server = https://mirror.msys2.org/mingw/ucrt64/
# Server = https://repo.msys2.org/mingw/ucrt64/

## Tier 1
# Server = https://mirror.umd.edu/msys2/mingw/ucrt64/
# Server = https://mirror.yandex.ru/mirrors/msys2/mingw/ucrt64/
# Server = https://download.nus.edu.sg/mirror/msys2/mingw/ucrt64/
# Server = https://mirror.accum.se/mirror/msys2.org/mingw/ucrt64/
# Server = https://ftp.nluug.nl/pub/os/windows/msys2/builds/mingw/ucrt64/
# Server = https://ftp.osuosl.org/pub/msys2/mingw/ucrt64/
# Server = https://mirror.internet.asn.au/pub/msys2/mingw/ucrt64/
# Server = https://mirror.selfnet.de/msys2/mingw/ucrt64/
# Server = https://mirrors.dotsrc.org/msys2/mingw/ucrt64/
# Server = https://mirrors.bfsu.edu.cn/msys2/mingw/ucrt64/
# Server = https://mirrors.tuna.tsinghua.edu.cn/msys2/mingw/ucrt64/
Server = https://mirrors.ustc.edu.cn/msys2/mingw/ucrt64/
# Server = https://mirror.nju.edu.cn/msys2/mingw/ucrt64/
# Server = https://repo.extreme-ix.org/msys2/mingw/ucrt64/
# Server = https://mirror.clarkson.edu/msys2/mingw/ucrt64/
# Server = https://quantum-mirror.hu/mirrors/pub/msys2/mingw/ucrt64/
# Server = https://mirror.archlinux.tw/MSYS2/mingw/ucrt64/
# Server = https://fastmirror.pp.ua/msys2/mingw/ucrt64/

## Tier 2
# Server = https://ftp.cc.uoc.gr/mirrors/msys2/mingw/ucrt64/
# Server = https://mirror.jmu.edu/pub/msys2/mingw/ucrt64/
# Server = https://mirrors.piconets.webwerks.in/msys2-mirror/mingw/ucrt64/
# Server = https://www2.futureware.at/~nickoe/msys2-mirror/mingw/ucrt64/
# Server = https://mirrors.sjtug.sjtu.edu.cn/msys2/mingw/ucrt64/
# Server = https://mirrors.bit.edu.cn/msys2/mingw/ucrt64/
# Server = https://mirrors.aliyun.com/msys2/mingw/ucrt64/
# Server = https://mirror.iscas.ac.cn/msys2/mingw/ucrt64/
# Server = https://mirrors.cloud.tencent.com/msys2/mingw/ucrt64/
```

只保留 `https://mirrors.ustc.edu.cn/` 开头的（即科大源），其他行都用 `#` 注释掉，如上所示。

其他的镜像源文件以此重复上述注释就行，改完以后要使修改生效，需执行：

```bash
pacman -Syyu
```

该命令的作用是：同步更新所有环境的软件包数据库。

### 3.3 MSYS2 中的环境

当安装完 MSYS2 后，发现 MSYS2 带有不同的后缀，如：CLANG64、CLANG32、CLANGARM64、MINGW32、MINGW64、MSYS、UCRT64 等，默认使用的是 `UCRT64` 模式。

MSYS2 提供了不同的环境/子系统，首先用户需要决定要使用哪个环境。

这些环境之间的区别主要在于环境变量、默认编译器/链接器、架构、使用的系统库等方面。

- environment variables：环境变量
- default compilers/linkers：默认编译器/链接器
- architecture：架构
- system libraries used etc：使用的系统库

如果不确定，建议选择 UCRT64 环境。

**环境对照表**

MSYS 环境包含基于类 Unix/cygwin 的工具，存储在 /usr 目录下，并且它是特殊的，因为它始终处于活动状态。所有其他环境都继承自 MSYS 环境并在其基础上添加各种功能。

例如，在 UCRT64 环境中，$PATH 变量以 /ucrt64/bin:/usr/bin 开头，因此可以使用所有 /ucrt64 和 /msys 目录下的工具。

各环境的细节一览表：

| Name | Prefix | Toolchain | Architecture | C Library | C++ Library |
|:-----|:-------|:----------|:-------------|:----------|:------------|
| MSYS(*) | /usr | gcc | x86_64 | cygwin | libstdc++ |
| UCRT64 | /ucrt64 | gcc | x86_64 | ucrt | libstdc++ |
| CLANG64 | /clang64 | llvm | x86_64 | ucrt | libc++ |
| CLANGARM64 | /clangarm64 | llvm | aarch64 | ucrt | libc++ |
| CLANG32 | /clang32 | llvm | i686 | ucrt | libc++ |
| MINGW64 | /mingw64 | gcc | x86_64 | msvcrt | libstdc++ |
| MINGW32 | /mingw32 | gcc | i686 | msvcrt | libstdc++ |

### 3.4 切换环境

MSYS 2.0 安装好以后，系统开始菜单里 `MSYS 2.0` 下面会有不同环境的快捷方式，例如：`MSYS2 UCRT64.lnk`、`MSYS2 MINGW64.lnk` 等等，假如找不到快捷方式，也可以通过如下方式打开不同的环境：

例如，你的 MSYS 2.0 的安装目录是：C:\msys64。

- MSYS2 UCRT64: C:\msys64\ucrt64.exe
- MSYS2 MINGW64: C:\msys64\mingw64.exe
- MSYS2 MSYS: C:\msys64\msys.exe
- MSYS2 CLANG64: C:\msys64\clang64.exe

以此类推，推荐的环境有 `UCRT64` 或 `MINGW64`，某些时候你的 VS 版本过低，可以考虑使用 `MINGW64` 环境，否则推荐使用 `UCRT64` 。

### 3.5 安装 mingw-w64

不同的环境使用不同的安装命令，以 `UCRT64` 环境为例，命令为：

```bash
pacman -S mingw-w64-ucrt-x86_64-gcc
```

类似的，`MINGW64` 环境的命令就是：`pacman -S mingw-w64-x86_64-gcc` 。

**安装包前缀对照表**：

| Name    | Package prefix |
|---------|----------------|
| MSYS    | None |
| MINGW32 | mingw-w64-i686- |
| MINGW64 | mingw-w64-x86_64- |
| UCRT64  | mingw-w64-ucrt-x86_64- |
| CLANG32 | mingw-w64-clang-i686- |
| CLANG64 | mingw-w64-clang-x86_64- |
| CLANGARM64 | mingw-w64-clang-aarch64- |

其他需要安装的工具，还有：

```bash
pacman -S git make cmake yasm
pacman -S --needed base-devel
pacman -S autoconf autogen pkg-config
pacman -S mingw-w64-ucrt-x86_64-gdb
pacman -S mingw-w64-ucrt-x86_64-nasm
pacman -S mingw-w64-ucrt-x86_64-diffutils
```

据说，mingw-w64-ucrt-x86_64-nasm 版本比默认的 nasm 性能更好，但要注意每个你要用的环境都要单独安装，diffutils 也类似。

其他环境下以此类推，例如：`MINGW32` 环境，命令如下：

```bash
pacman -S mingw-w64-i686-gdb
pacman -S mingw-w64-i686-nasm
pacman -S mingw-w64-i686-diffutils
```

环境的前缀对照表请参考上文。

### 3.6 mingw-w64 toolchain

安装 UCRT64 环境的 GCC 可以直接安装 mingw-w64-ucrt-x86_64-toolchain 这个包，这是 MSYS2 所定义的一个 Group，简单说就是一个包组，是包含 UCRT64 环境 C 编译器的软件包的一个组合包。包含了：

```bash
mingw-w64-ucrt-x86_64-binutils
mingw-w64-ucrt-x86_64-crt-git
mingw-w64-ucrt-x86_64-gcc
mingw-w64-ucrt-x86_64-gcc-ada
mingw-w64-ucrt-x86_64-gcc-fortran
mingw-w64-ucrt-x86_64-gcc-libgfortran
mingw-w64-ucrt-x86_64-gcc-libs
mingw-w64-ucrt-x86_64-gcc-objc
mingw-w64-ucrt-x86_64-libgccjit
mingw-w64-ucrt-x86_64-gdb
mingw-w64-ucrt-x86_64-gdb-multiarch
mingw-w64-ucrt-x86_64-headers-git
mingw-w64-ucrt-x86_64-libmangle-git
mingw-w64-ucrt-x86_64-libwinpthread-git
mingw-w64-ucrt-x86_64-winpthreads-git
mingw-w64-ucrt-x86_64-make
mingw-w64-ucrt-x86_64-pkgconf
mingw-w64-ucrt-x86_64-tools-git
mingw-w64-ucrt-x86_64-winstorecompat-git
```

安装命令是：

```bash
pacman -S mingw-w64-ucrt-x86_64-toolchain
```

toolchain 安装包的好处安装得更完整，但是同时，也会安装一些没用的安装包，占用的磁盘空间会更大。

**toolchain 安装包名的对照表**：

| Name    | Package prefix |
|---------|----------------|
| MSYS    | None |
| MINGW32 | mingw-w64-i686-toolchain |
| MINGW64 | mingw-w64-x86_64-toolchain |
| UCRT64  | mingw-w64-ucrt-x86_64-toolchain |
| CLANG32 | mingw-w64-clang-i686-toolchain |
| CLANG64 | mingw-w64-clang-x86_64-toolchain |
| CLANGARM64 | mingw-w64-clang-aarch64-toolchain |

### 3.7 mingw-w64 环境变量

例如，你的 MSYS 2.0 的安装路径是 `C:\msys64`，使用的是 `UCRT64` 环境，则把 `C:\msys64\ucrt64\bin` 目录添加到系统的 `Path` 中。

此外，把 MSYS 2.0 的 bin 目录 `C:\msys64\usr\bin` 也加入到系统的环境变量 `Path` 中。这是让 Windows 上其他命令行终端也可以使用 MSYS 2.0 和 mingw-w64 的命令。

打开一个系统自带的命令行终端，输入 `gcc -v`，如果可以查看如下的 gcc 版本信息，则说明环境变量配置成功了。

如下所示：

```bash
$ gcc -v

Using built-in specs.
COLLECT_GCC=C:\msys64\ucrt64\bin\gcc.exe
COLLECT_LTO_WRAPPER=C:/msys64/ucrt64/bin/../lib/gcc/x86_64-w64-mingw32/14.2.0/lto-wrapper.exe
Target: x86_64-w64-mingw32
Configured with: ../gcc-14.2.0/configure --prefix=/ucrt64 \
--with-local-prefix=/ucrt64/local --build=x86_64 \
-w64-mingw32 --host=x86_64-w64-mingw32 --target=x86_64-w64-mingw32 \
--with-native-system-header-dir=/ucrt64
/include --libexecdir=/ucrt64/lib --enable-bootstrap --enable-checking=release \
--with-arch=nocona --with-tune=generic \
--enable-languages=c,lto,c++,fortran,ada,objc,obj-c++,rust,jit \
--enable-shared --enable-static --enable-libatomic --enable-threads=posix \
--enable-graphite --enable-fully-dynamic-string --enable-libstdcxx-filesystem-ts \
--enable-libstdcxx-time --disable-libstdcxx-pch --enable-lto --enable-libgomp \
--disable-libssp --disable-multilib --disable-rpath --disable-win32-registry \
--disable-nls --disable-werror --disable-symvers --with-libiconv --with-system-zlib \
--with-gmp=/ucrt64 --with-mpfr=/ucrt64 --with-mpc=/ucrt64 --with-isl=/ucrt64 \
--with-pkgversion='Rev2, Built by MSYS2 project' \
--with-bugurl=https://github.com/msys2/MINGW-packages/issues --with-gnu-as --with-gnu-ld \
--disable-libstdcxx-debug --enable-plugin --with-boot-ldflags=-static-libstdc++ \
--with-stage1-ldflags=-static-libstdc++
Thread model: posix
Supported LTO compression algorithms: zlib zstd
gcc version 14.2.0 (Rev2, Built by MSYS2 project)
```

这里可以看到，gcc 的版本是 `14.2.0` 。

## 4. 手动安装 mingw-w64

有些时候，如果你觉得 `MSYS 2.0` 自带的 mingw-w64 版本太旧或太新，你可以选择自己手动安装 mingw-w64。

首先，进去官网地址：[https://www.mingw-w64.org](https://www.mingw-w64.org)，左侧点击 “Downloads”，然后在 Downloads 页面是右侧，找到 “MinGW-W64-builds” 并点击，会跳转到如下内容：

**MinGW-W64-builds**

Installation: [GitHub](https://github.com/niXman/mingw-builds-binaries/releases)

**MSYS2**

Installation: [GitHub](http://msys2.github.io/)

点击上面第一个 GitHub 的链接：[https://github.com/niXman/mingw-builds-binaries/releases](https://github.com/niXman/mingw-builds-binaries/releases)

这里选择 `x86_64-14.2.0-release-posix-seh-msvcrt-rt_v12-rev0.7z` 版本。

注意：这里跟上一节的选择环境不一样，选择的是 `MINGW64` (msvcrt) 模式。

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

下载完成后，把压缩包里的目录和文件解压到 `C:\msys64` 目录下，对应的环境的 mingw-w64 目录下，例如：这里选择的是 `MINGW64` 环境，则它的 mingw-w64 环境目录是：C:\msys64\mingw64，用压缩包的文件覆盖这个文件夹的内容即可。

**gcc版本**

检查 gcc 的版本，可以看到也是 `14.2.0`，但编译选项跟 MSYS 2.0 自带的 `14.2.0` 是有点不一样的。

```bash
$ gcc -v

Using built-in specs.
COLLECT_GCC=C:\msys64\mingw64\bin\gcc.exe
COLLECT_LTO_WRAPPER=C:/msys64/mingw64/bin/../libexec/gcc/x86_64-w64-mingw32/14.2.0/lto-wrapper.exe
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

## 5. 参考文章

- [Windows10 安装 MSYS2](https://muxiner.github.io/windows10-msys2-installation/)

- [windows上安装mingw教程及mingw64国内下载地址汇总](https://blog.csdn.net/FL1623863129/article/details/142673029)

- [【软件教程】MingW-W64-builds不同版本之间的区别](https://blog.csdn.net/zhangjiuding/article/details/129556458)

- [史上最全msys2下载配置操作步骤](https://blog.csdn.net/xuxu_123_/article/details/136574282)

- [超详细教程：windows安装MSYS2（mingw && gcc）](https://blog.csdn.net/ymzhu385/article/details/121449628)

- [Windows下使用MinGW+msys编译ffmpeg](https://www.cnblogs.com/shines77/p/3500337.html)

- [MSYS2 官网](https://www.msys2.org/)
