# Windows下编译FFmpeg（三）-- 依赖库的下载与安装

## 导航

- [Windows下编译FFmpeg（一）-- MSYS2 和 mingw-w64](./Windows下编译FFmpeg（一）--MSYS2和mingw-w64.md)
- [Windows下编译FFmpeg（二）-- MinGW32 和 msys 1.0](./Windows下编译FFmpeg（二）--MinGW32和msys-1.0.md)
- [Windows下编译FFmpeg（三）-- 依赖库的下载与安装](./Windows下编译FFmpeg（三）--依赖库的下载与安装.md)
- [Windows下编译FFmpeg（四）-- FFmpeg 编译选项](./Windows下编译FFmpeg（四）--FFmpeg编译选项.md)

## 1. 前言

FFmpeg 支持大量的外部库，其中一部分是增加功能的，一部分是增强性能的，一部分是支持特定硬件、软件环境的，还有些跟 FFmpeg 内置功能有重复，但是外部库可以提供更高品质或更多选项。

用./configure --help可以显示出它支持的所有外部库，每个外部库的名字和说明右边还有一个方括号，里面是 [no] 或者 [autodetect]。no 的意思是默认不安装，必须明确写了 --enable 选项才安装；autodetect 的意思是只要系统里已经安装了这个库，就会被包含进去，除非明确写了 --disable 选项来排除它。

要注意，一个外部库能否被包含进 FFmpeg 的最终程序，也受配置时选择的授权协议限制，如果选择了默认的 LGPL 授权协议，那么 libx264 之类 GPL 授权的外部库就会被排除掉。

有些外部库功能类似但互相抵触，只能选择一个，例如：openssl, gnutls, mbedtls 就只能选择其一，它们都是提供 tls 网络通信支持的。其实我们为 Windows 编译 FFmpeg 的时候，这三个都不需要，因为 Windows 上默认使用 schannel，而 schannel 甚至不需要用 pacman 安装，它是 ucrt64 环境自带的。

## 2. 依赖库

### 2.1 相关知识

自己编译 FFmpeg 的时候，一般不建议把所有外部库都包含进去，很多是用不上的，有些在当下的使用环境里毫无意义，比如，alsa 是 Linux 专用的，AppKit 是 MacOS 专用的，libdc1394 是用来读取 1394 接口数字摄像机的，早就过时了，xvid 并不比当前 FFmpeg 内置的 MPEG-4 编解码器更好，通常人们并不用 FFmpeg 通过 openjpeg 库来处理 jpeg2000 静态图像。总之，如果知道用不上，或者搞不懂是一个外部库是干什么的，也不知道怎么使用，就不要选。

**WebM格式**
较新的 WebM 格式，该格式的音频编码主要采用 Opus，视频编码采用 VP8 或者 VP9。其中 Opus 的编解码库为 libopus，VP8 和 VP9 的编解码库为 libvpx 。

### 2.1 源安装

#### 2.1.1 mingw-w64

**SDL2**

通过 pacman 安装 SDL2 ：

```bash
pacman -S mingw-w64-ucrt-x86_64-SDL2
```

需要添加编译开关 `--enable-sdl2` 。

**ffnvcodec**

给 FFmpeg 加上 nVidia 硬件编解码器的支持。通过 pacman 安装 ffnvcodec ：

```bash
pacman -S  ucrt64/mingw-w64-ucrt-x86_64-ffnvcodec-headers
```

`./configure` 的选项不用修改，因为跟 nVidia 硬件相关的库都是自动检测的。这个库只需要一些 .h 文件，因为真正运行的代码是随着 nVidia 的驱动程序安装的几个 DLL，并且 FFmpeg 使用动态加载的方式寻找和使用这些 DLL，所以连导入库也不需要，如果系统里没有安装这些 DLL，FFmpeg 会在被要求使用相关功能时报错，而不会一启动就因为缺少 DLL 而无法运行。

`./configure` 完以后，会看到多了几个跟 nVidia 相关的模块。

```bash
External libraries providing hardware acceleration:
cuda                    d3d11va                 dxva2                   nvdec
cuvid                   d3d12va                 ffnvcodec               nvenc
```

**OneVPL**

oneVPL 视频库属于英特尔 oneAPI 工具包的一部分。这是一个完整的视频处理库，集成了视频编码、解码以及后处理功能 (Post Processing)。如果支持 oneVPL 接口，oneVPL 视频库允许构建可在 CPU、GPU 和其他加速器上执行的可移植多媒体管道 (portable media pipelines)。

oneVPL 的仓库提供了进一步的介绍：

> 它在以媒体为中心的 workload 和视频分析 workload 中提供设备发现和选择，并为零拷贝缓冲区共享提供 API 原语。oneVPL 是向后和跨架构兼容的，可确保在当前和下一代硬件上实现最佳执行，而无需更改源代码。

目前，英特尔提供了一个基于 CPU 的 oneVPL 后端，以及一个针对 Gen12 / Xe 图形和更新的原生 oneVPL 实现。同时，对于较旧的英特尔硬件，他们具有英特尔媒体 SDK 集成，能够使用支持现代 oneVPL 接口的软件，可用于旧驱动程序。

这个库在 2022 年 8月左右才由 Intel 提交到 FFmpeg，所以低于或等于 5.1.6 (2022-07-13) 的版本不支持（6.0.1 - 2023-02-19）。需要安装 oneVPL 2.0 或更新版本，并且不能与 FFmpeg 旧的 Intel Media SDK / Quick Sync Video (QSV) 一起作为同一构建的一部分。

以下是 Intel Media SDK / Quick Sync Video (QSV) 的信息：

```bash
--enable-libmfx          enable Intel MediaSDK (AKA Quick Sync Video) code via libmfx [no]
--disable-vaapi          disable Video Acceleration API (mainly Unix/Intel) code [autodetect]
```

INTEL 的 CPU 里面也有一套硬件视频编解码器，它用到的库叫做 libvpl 。我们先安装 libvpl ：

```bash
pacman -S ucrt64/mingw-w64-ucrt-x86_64-libvpl

pacman -S ucrt64/mingw-w64-ucrt-x86_64-libvpx
pacman -S ucrt64/mingw-w64-ucrt-x86_64-libopus   ## 没有
pacman -S ucrt64/mingw-w64-ucrt-x86_64-libmfx

pacman -S ucrt64/mingw-w64-ucrt-x86_64-fdk-aac
pacman -S ucrt64/mingw-w64-ucrt-x86_64-libx264

# 包含 x264, x265 等等很多包
pacman -S ucrt64/mingw-w64-ucrt-x86_64-x264
```

需要添加编译开关 `--enable-libvpl` 。

其他依赖库:

- zlib
- mp3lame
- xvidcore
- libogg
- libvorbis
- faad
- faac
- amr_nb
- amr_wb
- libdts
- libgsm
- x264

## 2. 编译安装

**libopus**

Opus 是一种处理语音交互和音频传输的编码标准，该标准的编解码器叫做 libopus 。添加编译选项 `--enable-libopus` 。

下载地址：[https://ftp.osuosl.org/pub/xiph/releases/opus/](https://ftp.osuosl.org/pub/xiph/releases/opus/)

2024年12月最新的版本是 lbopus-1.5.2 。下载地址如下：

```bash
https://ftp.osuosl.org/pub/xiph/releases/opus/opus-1.5.2.tar.gz
```

执行以下命令解压：

```bash
tar xzvf opus-1.5.2.tar.gz
cd opus-1.5.2
```

配置，编译并安装：

```bash
./configure --prefix=/usr/local/libopus
make
make install
```

给环境变量 `PKG_CONFIG_PATH` 添加 libopus 的 pkgconfig 路径，也就是在 `/etc/profile` 文件末尾添加如下一行内容：

```bash
export PKG_CONFIG_PATH=/usr/local/libopus/lib/pkgconfig:$PKG_CONFIG_PATH

## 重新加载 /etc/profile, 让环境变量生效
source /etc/profile

## 检查环境变量
env | grep PKG_CONFIG_PATH
```

**libvpx**

libvpx 是视频编码标准 VP8 和 VP9 的编解码器。添加编译选项 `--enable-libvpx` 。

下载地址：[https://github.com/webmproject/libvpx/tags](https://github.com/webmproject/libvpx/tags)

2024年12月最新的版本是 libvpx-1.15.0 。下载地址如下：

```bash
https://github.com/webmproject/libvpx/archive/refs/tags/v1.15.0.tar.gz
```

执行以下命令解压：

```bash
tar xzvf libvpx-1.15.0.tar.gz
cd libvpx-1.15.0
```

这里注意，如果在 MinGW 环境下编译 libvpx 要用到 yasm 编译汇编代码，这里不能使用 Windows 版的 yasm 版本，请使用 `which yasm` 查看路径，并改回 MinGW 版本的 yasm 。或者使用 `--as=nasm` 编译选项，选择 `nasm` 编译汇编代码。

配置，编译并安装：

（如果不加 `--enable-pic`，在编译 FFmpeg 时会报错“relocation R_X86_64_32 against `.rodata.str1.1' can not be used when making a shared object; recompile with -fPIC”。）

```bash
./configure --prefix=/usr/local/libvpx --enable-static --enable-libyuv --enable-pic --as=nasm --disable-examples --disable-unit-tests
make
make install
```

`--enable-vp8 --enable-vp9 --enable-libyuv` 保留选项。

其中 `--enable-shared` 只支持 ELF, OS/2, and Darwin 系统。

**报错**

编译 `1.15.0 ` 版时报错，改用 `1.13.1` 版：

```bash
    [CXX] vp9/ratectrl_rtc.cc.o
make[1]: *** [Makefile:188：vp9/ratectrl_rtc.cc.o] 错误 1
make: *** [Makefile:17：.DEFAULT] 错误 2

```

给环境变量 `PKG_CONFIG_PATH` 添加 libvpx 的 pkgconfig 路径，也就是在 `/etc/profile` 文件末尾添加如下一行内容：

```bash
export PKG_CONFIG_PATH=/usr/local/libvpx/lib/pkgconfig:$PKG_CONFIG_PATH

## 重新加载 /etc/profile, 让环境变量生效
source /etc/profile

## 检查环境变量
env | grep PKG_CONFIG_PATH
```

## 3. NVENC 的支持

在 Windows 上编译 FFmpeg 并启用 NVENC 支持，需要确保正确配置 NVIDIA 的硬件加速库（nv-codec-headers）和 CUDA 工具包。以下是详细的步骤：

### 3.1 安装 CUDA Toolkit

NVENC 依赖于 NVIDIA 的 [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) 。

- 下载并安装 CUDA Toolkit：[CUDA Toolkit 下载页面](https://developer.nvidia.com/cuda-downloads) 。

- 安装时，确保选择将 CUDA 添加到系统环境变量（PATH）。

首先，要注意一点，`CUDA Toolkit` 是跟你的显卡驱动息息相关的，`CUDA Toolkit` 和显卡驱动的对应关系，可以查看这里：

[https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)

根据这个对照表，选择最新且与你的显卡驱动匹配的版本。旧版本的 `CUDA Toolkit` 在这里可以找到：

[https://developer.nvidia.com/cuda-toolkit-archive](https://developer.nvidia.com/cuda-toolkit-archive)

### 3.2 下载 Video Codec SDK

Video Codec SDK 跟下面的 `nv-codec-headers`，后者专门为 `MinGW` 环境特别修改过，但没有包含 `lib` 文件。

下载地址：[https://developer.nvidia.com/video-codec-sdk](https://developer.nvidia.com/video-codec-sdk)

旧版本的下载地址是：[https://developer.nvidia.com/video-codec-sdk-archive](https://developer.nvidia.com/video-codec-sdk-archive)

例如，我选择的是：`Video_Codec_SDK 12.1.14` 版本。

另外，经过实践，发现 FFmpeg 跟 Video Codec SDK 的版本之间有一定的关联性，因为新版的 `Video Codec SDK` 中废弃了很多 preset，参数 之类，所以旧版本的 FFmpeg 用太新的 Video Codec SDK 可能会报错。

例如，我用的 `FFmpeg 4.2.10` 版，最高只能适配到 `Video_Codec_SDK 11.1.5` ，再高就报错了。两者关系可以根据报错信息，搜索相应版本的源码中是否有这个 GUID 或 Enum 定义即可知晓。

这里有一个范例仓库：[Video Codec SDK Samples](https://github.com/NVIDIA/video-sdk-samples)

### 3.3 下载并配置 nv-codec-headers

nv-codec-headers 是 NVIDIA 提供的用于硬件加速编码的头文件。

**下载 nv-codec-headers**

```bash
git clone https://git.videolan.org/git/ffmpeg/nv-codec-headers.git
cd nv-codec-headers
```

由于 nv-codec-headers 的版本需要与显卡驱动版本相匹配，你可能需要手动下载特定版本的 nv-codec-headers，并修改其中的 ffnvcodec.pc 文件以适配你的系统。

例如，最新版的 `nv-codec-headers.git` 的 `README.md` 中这样写的：

```bash
FFmpeg version of headers required to interface with Nvidias codec APIs.

Corresponds to Video Codec SDK version 12.0.16.

Minimum required driver versions:
Linux: 550.54.14 or newer
Windows: 551.76 or newer
```

可以到 [https://git.videolan.org/](https://git.videolan.org/) 找到 `ffmpeg/nv-codec-headers.git` 并查询适合你的版本。

例如，我的显卡驱动的版本是 `537.13`，安装日期是 `2023/08/22`，那看 git 上的更新日期，在检查一下 `README.md` 中的版本号适合相符，所以我找到适合是：`n12.1.14.0`，导出这个版本即可。

**安装 nv-codec-headers**

在 nv-codec-headers 目录中，运行以下命令，并会显示如下内容：

```bash
$ make install

sed 's#@@PREFIX@@#C:/msys64/usr/local#' ffnvcodec.pc.in > ffnvcodec.pc
install -m 0755 -d '/usr/local/include/ffnvcodec'
install -m 0644 include/ffnvcodec/*.h '/usr/local/include/ffnvcodec'
install -m 0755 -d '/usr/local/lib/pkgconfig'
install -m 0644 ffnvcodec.pc '/usr/local/lib/pkgconfig'
```

可以看到，它想安装到 `C:/msys64/usr/local` 这个目录，但 `C:/msys64/usr` 目录下并没有 `local` 子目录，所以它安装到了 MSYS 2.0 默认的系统目录 `/usr/local/include` 中了，所以我们可以给它指定 `PREFIX` 安装路径为 `C:/msys64/ucrt64` (物理路径)，也就是绝对路径 `/ucrt64`，例如：

```bash
$ make PREFIX=/ucrt64 install

sed 's#@@PREFIX@@#C:/msys64/ucrt64#' ffnvcodec.pc.in > ffnvcodec.pc
install -m 0755 -d 'C:/msys64/ucrt64/include/ffnvcodec'
install -m 0644 include/ffnvcodec/*.h 'C:/msys64/ucrt64/include/ffnvcodec'
install -m 0755 -d 'C:/msys64/ucrt64/lib/pkgconfig'
install -m 0644 ffnvcodec.pc 'C:/msys64/ucrt64/lib/pkgconfig'
```

这次就正常了。

**FFmpeg配置时的问题**

FFmpeg 在配置过程中会生成这样的测试代码：

```cpp
#include <ffnvcodec/nvEncodeAPI.h>
NV_ENCODE_API_FUNCTION_LIST flist;
void f(void) { struct { const GUID guid; } s[] = { { NV_ENC_PRESET_HQ_GUID } }; }
int main(void) { return 0; }
```

其中 `NV_ENC_PRESET_HQ_GUID` 这个值在 `ffnvcodec` 中的头文件是没有的，导致识别不了 `--enable-nvenc`。可以自己添加到 `/ffnvcodec/nvEncodeAPI.h` 文件中，大约 244 行左右，写在 `NV_ENC_PRESET_P7_GUID` 这个值的后面，加入自己伪造的 `NV_ENC_PRESET_HQ_GUID` ，即可检测通过了。

```cpp
// {B974B872-2CE6-7D69-9EAE-4BC9016D0798}
static const GUID NV_ENC_PRESET_HQ_GUID  =
{ 0x974B872, 0x2CE6, 0x7D69, { 0x9E, 0xAE, 0x4B, 0xC9, 0x01, 0x6D, 0x07, 0x98 } };
```

**FFmpeg编译选项**

```bash
./configure \
    --enable-gpl \
    --enable-nonfree \
    --enable-cuda \
    --enable-cuvid \
    --enable-nvenc \
    --enable-nvdec \
    --enable-libnpp \
    --enable-shared \
    --enable-avisynth \
    --enable-ffnvcodec \
    --extra-cflags="-I/usr/local/cuda/include" \
    --extra-ldflags="-L/usr/local/cuda/lib64"
```

**参数说明**

- **--enable-cuda**：启用 CUDA 支持。

- **--enable-cuvid**：启用 NVIDIA CUVID 硬件解码支持。

- **--enable-nvenc**：启用 NVENC 硬件编码支持。

- **--enable-nvdec**：启用 NVDEC 硬件解码支持。

- **--enable-libnpp**：启用 NVIDIA Performance Primitives (NPP) 库支持。

- **--extra-cflags** 和 **--extra-ldflags**：指定 CUDA 头文件和库的路径。

## 4. 常用依赖库

**libjpeg**

因为 libyuv 依赖于 libjpeg，所以得先编译它，我们可以选择在 mingw-w64 环境下编译。

由于编译过程可能会用到 yasm 汇编，前面的文章有介绍过，如何在 mingw-w64 环境下安装 yasm，最好使用 Windows 版的 yasm。

libjpeg-turbo 下载：

```bash
git clone https://github.com/libjpeg-turbo/libjpeg-turbo.git
```

MinGW 环境下编译：

```bash
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX="../build_release" -DCMAKE_BUILD_TYPE="Release" -G "MSYS Makefiles" ..
cmake --build . --config Release
cmake --build . --target install --config Release
```

这里我们把安装路径设置为 `../build-release`，待会有用，如果这个目录不会自动创建，在安装之前可以自己创建一下这个目录，也就是跟 `build` 目录同一级下新建一个 `build-release` 目录。

编译和安装完以后，你会看到 `build-release` 目录下面创建了 bin, lib, include, share 等目录。

给你的 Windows 添加两个系统环境变量，分别是 `JPEG_INCLUDE_DIR` 和 `JPEG_LIBRARY` 分别指向 `build-release` 目录下的 `include` 和 `lib` 目录。这在待会编译 `libyuv` 的时候，cmake 要检测这两个路径判断是否安装了 libjpeg。

如果你也想把编译好的 libjpeg 安装到你的 mingw-w64 环境中，使用下面的 cmake 命令重新构建：

```bash
cmake -DCMAKE_INSTALL_PREFIX="/ucrt64" -DCMAKE_BUILD_TYPE="Release" -G "MSYS Makefiles" ..
```

然后象上面的命令一样，编译和安装，即会自动安装到 mingw-w64 所在的 MSYS 2.0 中。

这里的安装路径不要写 "/ucrt64" 而不是 "/ucrt64/lib" 的原因，前面你也看到登录，他会自动在 `/ucrt64` 后面加 lib, bin, include 路径。

由于 `mingw-w64` 可以通过 pacman 命令直接安装 libjpeg，所以其实这一步是可以不做的。

**libyuv**

libyuv 是 Google 开源的实现各种 YUV 与 RGB 之间相互转换、旋转、缩放的库。

官网：[https://chromium.googlesource.com/libyuv/libyuv](https://chromium.googlesource.com/libyuv/libyuv)

非官方镜像：[https://github.com/lemenkov/libyuv](https://github.com/lemenkov/libyuv)

```bash
git clone https://chromium.googlesource.com/libyuv/libyuv
```

由于我尝试了好多次，发现都不能在 mingw-w64 环境下编译 libyuv，所以在 Windows 上建议直接用 CMake-GUI 构建 libyuv，在 CMake-GUI 上选择 libyuv 的路径，生成路径设为 libyuv 根目录下的 `build_msvc`，点 `Configure` 按钮，成功后再点 `Generate` 按钮即可。

如果按上面的配置好了 libjpeg，它会自己找到 libjpeg 所在的路径，用 MSVC 打开生成的 YUV.sln 文件，直接编译即可，没有遇到任何问题，简单快捷。

下面还是介绍了在 mingw-w64 环境下编译的过程，但 make 的时候是会报错的，有兴趣可以自己试一下。

**mingw-w64 下编译 libyuv**

例如：在 mingw-w64 下的 UCRT64 环境，这会影响 PREFIX 安装路径的设置，请根据具体情况自行修改。

编译和安装的命令如下：

```bash
mkdir build
cd build

cmake -DCMAKE_INSTALL_PREFIX="/ucrt64/lib" -DCMAKE_BUILD_TYPE="Release" -G "MSYS Makefiles" ..
cmake --build . --config Release
cmake --build . --target install --config Release
```

编译成功后，会在指定的安装路径下生成 libyuv.a 和 libyuv.so 文件‌。

`-DCMAKE_VERBOSE_MAKEFILE=ON` 。

**错误处理**

如果你在用 cmake 构建的时候报如下错误时：

```bash
CMake Error at /ucrt64/share/cmake/Modules/CMakeTestCXXCompiler.cmake:73 (message):
  The C++ compiler

    "/ucrt64/bin/g++.exe"

  is not able to compile a simple test program.
```

请在 CMakeLists.txt 中的 `project ( YUV C CXX )` 这句前面加入：

```cmake
INCLUDE(CMakeForceCompiler)
# 这里要写绝对路径，不能用相对路径，因为 CMake 识别不了 MSYS 2.0 的相对路径...
set(CMAKE_C_COMPILER "c:/msys64/ucrt64/bin/gcc.exe")
set(CMAKE_CXX_COMPILER "c:/msys64/ucrt64/bin/g++.exe")
```

并把 CMakeTestCXXCompiler.cmake，CMakeTestCCompiler.cmake 报错的语句相应的代码注释掉，即可。

但执行 `cmake --build . --config Release` 命令的时候会报错，无法完成编译。

## x. 参考文章

- [在windows上编译FFmpeg](https://zhuanlan.zhihu.com/p/707298876)

- [在Windows下编译ffmpeg完全手册](https://blog.51cto.com/u_15329201/3418475)

- [FFmpeg 初步支持 oneVPL](https://weibo.com/ttarticle/p/show?id=2309404803412083736718)

- [Windows下使用MinGW+msys编译ffmpeg](https://www.cnblogs.com/shines77/p/3500337.html)

- [Hardware/QuickSync - Intel Quick Sync Video](https://trac.ffmpeg.org/wiki/Hardware/QuickSync)

- [FFmpeg开发笔记（十三）Windows环境给FFmpeg集成libopus和libvpx](https://blog.csdn.net/aqi00/article/details/136945020)
