# Windows下编译FFmpeg（四）-- FFmpeg编译选项

## 导航

- [Windows下编译FFmpeg（一）-- MSYS2 和 mingw-w64](./Windows下编译FFmpeg（一）--MSYS2和mingw-w64.md)
- [Windows下编译FFmpeg（二）-- MinGW32 和 msys 1.0](./Windows下编译FFmpeg（二）--MinGW32和msys-1.0.md)
- [Windows下编译FFmpeg（三）-- 依赖库的下载与安装](./Windows下编译FFmpeg（三）--依赖库的下载与安装.md)
- [Windows下编译FFmpeg（四）-- FFmpeg 编译选项](./Windows下编译FFmpeg（四）--FFmpeg编译选项.md)

## 1. 前言

FFmpeg 是一个开源的多媒体框架，它包括了一套可以用来处理视频和音频数据的库（libav）以及一些命令行工具。FFmpeg 能够处理几乎所有格式的视频和音频文件，包括转换格式、剪辑、合并、过滤、编码和解码等操作。

FFmpeg 的主要组件包括：

- **libavcodec**：这是一个编解码器库，支持多种音视频编解码格式。
- **libavformat**：用于音视频封装格式处理的库。
- **libavfilter**: 是一个包含媒体过滤器的库。
- **libavutil**：提供一些公共的工具函数，比如数学运算、随机数生成等。
- **libavdevice**: 是一个包含输入和输出设备的库，用于从许多常见的多媒体输入/输出软件框架中抓取和渲染，包括 Video4Linux、Video4Linux2、VfW 和 ALSA。
- **libswscale**：用于执行高度优化的图像缩放和色彩空间/像素格式转换操作的库。
- **libswresample**：用于执行高度优化的音频重采样、重新矩阵和样本格式转换操作的库。从 FFmpeg 5.0 开始，这个库被移除。

FFmpeg 的命令行工具包括：

- **ffmpeg**：用于采集、编码、解码音视频，推送音视频流，以及音视频文件格式转换等。
- **ffplay**：一个简单的播放器，可以播放多媒体文件。
- **ffprobe**：用于显示多媒体文件信息的工具。

FFmpeg 因其强大的功能和灵活性而被广泛应用于视频网站、视频编辑软件、视频转换工具等多种场景。

## 2. FFmpeg 编译选项

### 2.1 基本编译开关

首先，可以增加一些常规编译选项来减小最终编译包的大小。可以使用 `./configure -h` 命令来列出 configure 程序支持的编译选项，每一项编译选项后都有对应的解释。

以下是一些常用的编译开关：

```bash
--enable-shared: 编译生成 dll 动态库版本
--enable-static: 编译生成静态库版本，这是默认值，可不写
--disable-static: 不生成静态库版本，由于 exe 是静态链接，使用了该选项则不会编译 exe
--cpu=i686: 选择最小要求的 CPU 类型 (影响指令选择, 可能会导致比较旧的CPU崩溃)
--arch=x86_32: x86_32 位版本
--arch=x86_64: x86_amd64 位版本
--host-os=win32: 当前 OS 类型, Windows 32 位系统
--host-os=win64: 当前 OS 类型, Windows 64 位系统
--target-os=win64: 目标 OS 类型, 不要加这个选项，设为 win64 可能会导致使用 msvc 的 lib.exe 来编译 dll
--disable-debug: 禁用 debugging 信息和符号
--enable-memalign-hack: 内存分配对齐 hack，这个开关已失效。
--extra-cflags=-I/mingw/include: include 目录
--extra-ldflags=-L/mingw/lib: lib 目录
--prefix=./build: 安装目录
--enable-asm: 允许编译 asm 代码
--enable-inline-asm: 允许编译内联 asm 代码
--toolchain=msvc: 设置工具链
--enable-cross-compile: 允许交叉编译
```

允许 GPL 3.0 协议的模块：

```bash
--enable-gpl --enable-version3 --enable-nonfree
```

禁用 GPL 协议的模块：

```bash
不带 --enable-gpl，且不带 --enable-nonfree 编译选项
```

### 2.2 常规编译开关

能够直接减小编译包大小的编译开关有如下几个：

```bash
--enable-small: 允许使用最小文件大小(MinReleaseSize)编译，而不是追求执行效率
--disable-doc: 禁止编译文档，可以避免将文档编译入包中
--disable-htmlpages: 禁止编译html文档，可以避免将文档编译入包中
--disable-manpages: 禁止编译man文档，可以避免将文档编译入包中
--disable-podpages: 禁止编译pod文档，可以避免将文档编译入包中
--disable-txtpages: 禁止编译txt文档，可以避免将文档编译入包中
--disable-runtime-cpudetect: 禁止运行时检测CPU性能，可以编出较小的包，这个不推荐
```

### 2.3 减少不必要的工具

最开始介绍了 ffmpeg 是基于 libav 开发的一套工具，除了 ffmpeg 之外，基于 libav 开发的工具还有：ffplay、ffprobe 以及 ffserver。这些不必要的工具是可以禁止掉的，相关选项为：

```bash
--disable-programs: 禁止编译命令行工具
--disable-ffmpeg: 禁止编译 ffmpeg 工具，这个不推荐
--disable-ffplay: 禁止编译 ffplay 工具
--disable-ffprobe: 禁止编译 ffprobe 工具
--disable-ffserver: 禁止编译 ffserver 工具，某些版本可能已移除
```

### 2.4 减少不必要的模块

```bash
libavcodec: --disable-avcodec
libavformat: --disable-avformat
libavfilter: --disable-avfilter
libswscale: --disable-swscale
libpostproc: --disable-postproc
libavdevice: --disable-avdevice
libswresample: --disable-swresample
libavresample: --enable-avresample
```

libav 包含以下几个模块：

- libavcodec: 该模块主要负责解码与编码，若无需该模块，可使用 --disable-avcodec 禁止编译，不过该模块为 libav 核心模块，非特殊情况最好不要禁止；

- libavformat: 该模块主要负责解封装与封装，若无需该模块，可使用 --disable-avformat 禁止编译，不过该模块为 libav 核心模块，非特殊情况最好不要禁止；

- libavfilter: 该模块主要负责音视频的过滤，包括裁剪、位置、水印等，若无需该模块，可使用 --disable-avfilter 禁止编译；

- libswscale: 该模块主要负责对原始视频数据进行场景转换、色彩映射以及格式转换，若无需该模块，可使用 --disable-swscale 禁止编译；

- libpostproc: 该模块主要负责对音视频进行后期处理，若无需该模块，可使用 --disable-postproc 禁止编译；

- libavdevice: 该模块主要负责与硬件设备的交互，若无需该模块，可使用 --disable-avdevice 禁止编译；

- libswresample: 该模块主要负责对原始音频数据进行格式转换，若无需该模块，可使用 --disable-swresample 禁止编译；

- libavresample: 该模块主要负责音视频封装编解码格式预设，该模块默认不编译，若要进行编译，使用 --enable-avresample 。FFmpeg 5.0 及以后的版本，已取消该模块。

### 2.5 减少不必要的设备

libav 可以从硬件设备中获取输入，同时也可以输出至硬件设备。可以指定支持的输入输出设备来避免不必要的编译：

```bash
--disable-devices: 禁止所有设备的编译
--disable-indevs: 禁止所有输入设备的编译
--disable-indev=NAME: 禁止特定输入设备的编译
--enable-indev=NAME: 允许特定输入设备的编译，搭配 –disable-indevs 可以实现单纯指定支持的输入设备
--disable-outdevs: 禁止所有输出设备的编译
--disable-outdev=NAME: 禁止特定输出设备的编译
--enable-outdev=NAME: 允许特定输出设备的编译，搭配 –disable-outdevs 可以实现单纯指定支持的输出设备
```

关于 libav 支持的输入输出设备名称，可以使用 `./configure --list-indevs` 和 `./configure --list-outdevs` 命令获取。

### 2.6 减少不必要的解析器

libav 可以对输入的数据进行格式检测，该功能由解析器 (parser) 负责。可以指定支持的解析器来避免不必要的编译：

```bash
--disable-parsers: 禁止所有解析器的编译
--disable-parser=NAME: 禁止特定解析器的编译
--enable-parser=NAME: 允许特定解析器的编译，搭配 --disable-parsers 可以实现单纯指定支持的解析器
```

关于 libav 支持的解析器名称，可以使用 `./configure --list-parsers` 命令获取。

### 2.7 减少不必要的二进制流过滤器

libav 可以将输入的数据转为二进制数据，同时可以对二进制数据进行特殊的处理，该功能由二进制流过滤器(bit stream filter)负责。可以指定支持的二进制流过滤器来避免不必要的编译：

```bash
--disable-bsfs: 禁止所有二进制流过滤器的编译
--disable-bsf=NAME: 禁止特定二进制流过滤器的编译
--enable-bsf=NAME: 允许特定二进制流过滤器的编译，搭配 –disable-bsfs 可以实现单纯指定支持的二进制流过滤器
```

关于 libav 支持二进制流过滤器名称，可以使用 `./configure --list-bsfs` 命令获取。

### 2.8 减少不必要的协议

libav 对于如何读入数据及输出数据制定了一套协议，同时 libav 内置了一些满足协议的方式，这些方式可以通过 `./configure --list-protocols` 列出。可以指定支持的输入输出方式来避免不必要的编译：

```bash
--disable-protocols: 禁止所有输入输出方式的编译
--disable-protocol=NAME: 禁止特定输入输出方式的编译
--enable-protocol=NAME: 允许特定输入输出方式的编译，搭配 –disable-protocols 可以实现单纯指定支持的输入输出方式
```

必须指定至少一种输入输出方式，通常通过使用 `--disable-protocols` 搭配 `--enable-protocol=NAME` 来完成。

### 2.9 减少不必要的组件

libav 处理音视频的流程中，负责解封装的是分离器 (demuxer)、负责封装的是复用器 (muxer)、负责音视频解码的为解码器 (decoder)、负责编码的为编码器 (encoder) 。

可以从 libav 所支持的四个组件的类型来减少不必要的编译。可以使用以下命令获取组件所支持的类型：

```bash
./configure --list-demuxers
./configure --list-muxers
./configure --list-decoders
./configure --list-encoders
```

（1）分离器：

```bash
--disable-demuxers: 禁止所有分离器的编译
--disable-demuxer=NAME: 禁止特定分离器的编译
--enable-demuxer=NAME: 允许特定分离器的编译，搭配 -–disable-demuxers
```

（2）复用器：

```bash
--disable-muxers: 禁止所有复用器的编译
--disable-muxer=NAME: 禁止特定复用器的编译
--enable-muxer=NAME: 允许特定复用器的编译，搭配 --disable-muxers
```

（3）解码器：

```bash
--disable-decoders: 禁止所有解码器的编译
--disable-decoder=NAME: 禁止特定解码器的编译
--enable-decoder=NAME: 允许特定解码器的编译，搭配 --disable-decoders
```

（4）编码器：

```bash
--disable-encoders: 禁止所有编码器的编译
--disable-encoder=NAME: 禁止特定编码器的编译
--enable-encoder=NAME: 允许特定编码器的编译，搭配 -–disable-encoders
```

至此，通过对项目的特殊定制，可以最大化的减小编译包的大小，避免编译包太大造成最终产品体积过大的问题。

## 3. 编译和使用

### 3.1 mingw-w64 + MSYS 2.0 + GCC

这种方式是完全只使用 mingw-w64 和 gcc，完全不需要 msvc 。

进入你的 FFmpeg 源码目录，例如：

```bash
cd /c/Project/OpenSrc/ffmpeg/ffmpeg-7.1
```

FFmpeg 7.1，编译成 dll，UCRT64 环境，LGPL 2.1：

```bash
./configure --enable-shared --disable-static --pkg-config-flags=--static \
--arch=x86_64 --host-os=win64 --disable-debug \
--extra-cflags=-I/ucrt64/include --extra-ldflags=-L/ucrt64/lib \
--prefix=./build_shared --enable-asm --enable-inline-asm \
--disable-doc --disable-htmlpages --disable-manpages --disable-podpages --disable-txtpages \
--enable-ffmpeg --disable-ffplay --disable-ffprobe \
--enable-avfilter --enable-avdevice --disable-swscale --disable-iconv \
--disable-decoders --enable-decoder=h264 --enable-decoder=hevc \
--enable-decoder=mpeg4 --enable-decoder=mjpeg --enable-decoder=aac \
--disable-encoders --enable-encoder=h264_nvenc --enable-encoder=hevc_nvenc \
--enable-encoder=mpeg4 --enable-encoder=mjpeg --enable-encoder=aac --enable-encoder=png \
--disable-demuxers --enable-demuxer=h264 --enable-demuxer=hevc \
--enable-demuxer=mpegvideo --enable-demuxer=mjpeg --enable-demuxer=aac \
--enable-demuxer=avi --enable-demuxer=mov --enable-demuxer=mpegps \
--disable-muxers --enable-muxer=h264 --enable-muxer=hevc \
--enable-muxer=mp4 --enable-muxer=mjpeg \
--enable-muxer=avi --enable-muxer=adts \
--disable-filters --enable-filter=fps --enable-filter=framerate \
--enable-filter=fsync --enable-filter=gblur --enable-bsfs \
--disable-protocols --enable-protocol=file --enable-protocol=http --enable-protocol=https \
--disable-parsers --enable-parser=h264 --enable-parser=hevc \
--enable-parser=mpeg4video --enable-parser=mjpeg --enable-parser=png \
--disable-indevs --enable-indev=gdigrab --enable-indev=vfwcap --enable-indev=dshow \
--disable-outdevs \
--enable-libvpl --enable-hardcoded-tables \
--enable-hwaccel=h264_nvdec --enable-hwaccel=h264_dxva2 \
--enable-hwaccel=hevc_nvdec --enable-hwaccel=hevc_dxva2 \
--disable-network
```

FFmpeg 7.1，编译成静态库，UCRT64 环境，LGPL 2.1：

```bash
./configure --enable-static --pkg-config-flags=--static \
--arch=x86_64 --host-os=win64 --disable-debug \
--extra-cflags=-I/ucrt64/include --extra-ldflags=-L/ucrt64/lib \
--prefix=./build_static --enable-asm --enable-inline-asm \
--disable-doc --disable-htmlpages --disable-manpages --disable-podpages --disable-txtpages \
--enable-ffmpeg --disable-ffplay --disable-ffprobe \
--enable-avfilter --enable-avdevice --disable-swscale --disable-iconv \
--disable-decoders --enable-decoder=h264 --enable-decoder=hevc \
--enable-decoder=mpeg4 --enable-decoder=mjpeg --enable-decoder=aac \
--disable-encoders --enable-encoder=h264_nvenc --enable-encoder=hevc_nvenc \
--enable-encoder=mpeg4 --enable-encoder=mjpeg --enable-encoder=aac --enable-encoder=png \
--disable-demuxers --enable-demuxer=h264 --enable-demuxer=hevc \
--enable-demuxer=mpegvideo --enable-demuxer=mjpeg --enable-demuxer=aac \
--enable-demuxer=avi --enable-demuxer=mov --enable-demuxer=mpegps \
--disable-muxers --enable-muxer=h264 --enable-muxer=hevc \
--enable-muxer=mp4 --enable-muxer=mjpeg \
--enable-muxer=avi --enable-muxer=adts \
--disable-filters --enable-filter=fps --enable-filter=framerate \
--enable-filter=fsync --enable-filter=gblur --enable-bsfs \
--disable-protocols --enable-protocol=file --enable-protocol=http --enable-protocol=https \
--disable-parsers --enable-parser=h264 --enable-parser=hevc \
--enable-parser=mpeg4video --enable-parser=mjpeg --enable-parser=png \
--disable-indevs --enable-indev=gdigrab --enable-indev=vfwcap --enable-indev=dshow \
--disable-outdevs \
--enable-libvpl --enable-hardcoded-tables \
--enable-hwaccel=h264_nvdec --enable-hwaccel=h264_dxva2 \
--enable-hwaccel=hevc_nvdec --enable-hwaccel=hevc_dxva2
```

FFmpeg 7.1，编译成 dll，UCRT64 环境，GPL 3.0：

```bash
./configure --enable-shared --disable-static --pkg-config-flags=--static \
--enable-gpl --enable-version3 --enable-nonfree \
--arch=x86_64 --host-os=win64 --disable-debug \
--extra-cflags=-I/ucrt64/include --extra-ldflags=-L/ucrt64/lib \
--prefix=./build_shared_gpl --enable-asm --enable-inline-asm \
--disable-doc --disable-htmlpages --disable-manpages --disable-podpages --disable-txtpages \
--enable-ffmpeg --disable-ffplay --disable-ffprobe \
--enable-avfilter --enable-avdevice --disable-swscale --disable-iconv \
--disable-decoders --enable-decoder=h264 --enable-decoder=hevc \
--enable-decoder=mpeg4 --enable-decoder=mjpeg --enable-decoder=aac \
--disable-encoders --enable-encoder=h264_nvenc --enable-encoder=hevc_nvenc \
--enable-encoder=mpeg4 --enable-encoder=mjpeg --enable-encoder=aac --enable-encoder=png \
--disable-demuxers --enable-demuxer=h264 --enable-demuxer=hevc \
--enable-demuxer=mpegvideo --enable-demuxer=mjpeg --enable-demuxer=aac \
--enable-demuxer=avi --enable-demuxer=mov --enable-demuxer=mpegps \
--disable-muxers --enable-muxer=h264 --enable-muxer=hevc \
--enable-muxer=mp4 --enable-muxer=mjpeg \
--enable-muxer=avi --enable-muxer=adts \
--disable-filters --enable-filter=fps --enable-filter=framerate \
--enable-filter=fsync --enable-filter=gblur --enable-bsfs \
--disable-protocols --enable-protocol=file --enable-protocol=http --enable-protocol=https \
--disable-parsers --enable-parser=h264 --enable-parser=hevc \
--enable-parser=mpeg4video --enable-parser=mjpeg --enable-parser=png \
--disable-indevs --enable-indev=gdigrab --enable-indev=vfwcap --enable-indev=dshow \
--disable-outdevs \
--enable-libvpl --enable-hardcoded-tables \
--enable-hwaccel=h264_nvdec --enable-hwaccel=h264_dxva2 \
--enable-hwaccel=hevc_nvdec --enable-hwaccel=hevc_dxva2
```

### 3.2 mingw-w64 + MSYS 2.0 + MSVC

这种方式特别的地方是，它是用 MSVC 的 cl.exe 来编译代码的。

先启动 MSVC 的命令行，然后再执行一个脚本跳转到 MSYS 2.0 Shell，这样就能继承 MSVC 命令行的设置。

#### 3.2.1 启动命令行

以 MSVC 2015 为例，从系统的开始菜单找到“Visual Studio 2015”一栏，在里面找到“VS2015 x64 本机工具命令提示符”，其他的命令行还有：

```bash
VS2015 x86 本机工具命令提示符
VS2015 x86 x64 兼容工具命令提示符
VS2015 x64 x86 兼容工具命令提示符
```

等等，不要弄错了，只有“VS2015 x64 本机工具命令提示符”是纯 64 位的命令行。

#### 3.2.2 msys2_shell.cmd

先把 MSYS 2.0 安装目录 C:\msys64 下的 msys2_shell.cmd 中的：

```bash
rem set MSYS2_PATH_TYPE=inherit
```

去掉前面的注释 "rem", 改成如下所示：

```bash
set MSYS2_PATH_TYPE=inherit
```

保存，退出。这样是为了将 vs 的环境继承给 MSYS2 。

然后，在 MSVC 2015 的命令行里执行如下命令：

```bash
C:\msys64\msys2_shell.cmd -ucrt64
```

这样就会跳出一个新的 MSYS2 的 shell 终端，该 shell 就继承了 vs2015 的环境路径。

你可以尝试在新的 MSYS2 shell 里输入：`$ echo $PATH`，将会看到继承了 vs2015 的 Path 设置。

启动参数和环境对照表：

| Name    | Parameter |
|---------|-----------|
| MSYS    | -msys 或 -msys2 |
| MINGW32 | -mingw32 |
| MINGW64 | -mingw64 |
| UCRT64  | -ucrt64 |
| CLANG32 | -clang32 |
| CLANG64 | -clang64 |
| CLANGARM64 | -clangarm64 |

另外，还有其他启动参数，一般无需设置，如下：

- **-mintty** ：启动 mintty 终端。
- **-conemu** ：启动 conemu 终端。
- **-defterm** ：启动 defterm 终端。

更多参数请自行阅读 msys2_shell.cmd 的源码。

#### 3.2.3 配置路径

在开始配置之前，我们还需要做一件事情：

```bash
$ which cl link yasm
```

检查一下这三个执行文件的路径对不对，如果是：

```bash
/c/Program Files (x86)/Microsoft Visual Studio/2017/Community/VC/Tools/MSVC/14.16.27023/bin/HostX64/x64/cl
/usr/bin/link
/usr/bin/yasm
```

把 `C:\msys64\usr\bin\link.exe` 改名为 `C:\msys64\usr\bin\link.exe.bak` ，

把 `C:\msys64\usr\bin\yasm.exe` 改名为 `C:\msys64\usr\bin\yasm.exe.bak` 。

yasm 1.3.0 版的 Windows 修正版可以到下面的网站下载：

- [vsyasm 1.3.0 2015-June-09 32bits for Visual Studio 2010, 2012 and 2013](http://www.megastormsystems.com/repository/Tools/yasm-1.3.0_2015-06-09_32bits.zip)
- [vsyasm 1.3.0 2015-June-09 64bits for Visual Studio 2010, 2012 and 2013](http://www.megastormsystems.com/repository/Tools/yasm-1.3.0_2015-06-09_64bits.zip)

把其中的 vsyasm.exe 拷贝到 `C:\msys64\ucrt64\bin` 目录下，并改名为 `yasm.exe` 。

yasm 官网的地址是：[http://yasm.tortall.net/Download.html](http://yasm.tortall.net/Download.html)，如果觉得上面的修正版不好用，也可以在官网下载。

再次检查路径：

```bash
$ which cl link yasm

/c/Program Files (x86)/Microsoft Visual Studio/2017/Community/VC/Tools/MSVC/14.16.27023/bin/HostX64/x64/cl
/c/Program Files (x86)/Microsoft Visual Studio/2017/Community/VC/Tools/MSVC/14.16.27023/bin/HostX64/x64/link
/ucrt64/bin/yasm
```

#### 3.2.4 编译选项

编译选项跟上一小节的 mingw-w64 + MSYS 2.0 + GCC 的差不多，主要的不同是可以加上 `--toolchain=msvc` 参数了，如下：

```bash
--toolchain=msvc --host-os=win64 --target-os=win64
```

除了以上参数，还需要修改 include 和 lib 的路径，找出你的 MSVC 版本的 cl.exe 到相应的 include、lib 目录的相对路径。

以 MSVC 2019 为例，相对路径为：

```bash
--extra-cflags=-I../../../../include --extra-ldflags=-L../../../../lib
```

进入你的 FFmpeg 源码目录，例如：

```bash
cd /c/Project/OpenSrc/ffmpeg/ffmpeg-7.1
```

**修改 config.h**

这里有个要注意的地方，在 `./configure` 配置完了以后，先别急着编译，先把新生成的 `config.h` 文件打开后保存为 UTF-8 格式。不做这一步在话，在 make 编译的时候，会出现如下无数的 warning，非常的烦人，烦还不是重点，一直显示 warning 会拖慢整个编译速度。可以用 UltraEdit 或者任何一款可以保存为 UTF-8 格式的编辑软件。

```bash
.\config.h(1): warning C4828: 文件包含在偏移 0x813 处开始的字符，该字符在当前源字符集中无效(代码页 65001)。
.\config.h(1): warning C4828: 文件包含在偏移 0x813 处开始的字符，该字符在当前源字符集中无效(代码页 65001)。
.\config.h(1): warning C4828: 文件包含在偏移 0x813 处开始的字符，该字符在当前源字符集中无效(代码页 65001)。
```

**编译选项**

FFmpeg 7.1，编译成 dll，UCRT64 环境，LGPL 2.1，MSVC 工具链：

```bash
./configure --enable-shared --disable-static --pkg-config-flags=--static \
--arch=x86_64 --toolchain=msvc --host-os=win64 --target-os=win64 --disable-debug \
--extra-cflags=-I../../../../include --extra-ldflags=-L../../../../lib \
--prefix=./build_msvc_shared --enable-asm --enable-inline-asm \
--disable-doc --disable-htmlpages --disable-manpages --disable-podpages --disable-txtpages \
--enable-ffmpeg --disable-ffplay --disable-ffprobe \
--enable-avfilter --enable-avdevice --disable-swscale --disable-iconv \
--disable-decoders --enable-decoder=h264 --enable-decoder=hevc \
--enable-decoder=mpeg4 --enable-decoder=mjpeg --enable-decoder=aac \
--disable-encoders --enable-encoder=h264_nvenc --enable-encoder=hevc_nvenc \
--enable-encoder=mpeg4 --enable-encoder=mjpeg --enable-encoder=aac --enable-encoder=png \
--disable-demuxers --enable-demuxer=h264 --enable-demuxer=hevc \
--enable-demuxer=mpegvideo --enable-demuxer=mjpeg --enable-demuxer=aac \
--enable-demuxer=avi --enable-demuxer=mov --enable-demuxer=mpegps \
--disable-muxers --enable-muxer=h264 --enable-muxer=hevc \
--enable-muxer=mp4 --enable-muxer=mjpeg \
--enable-muxer=avi --enable-muxer=adts \
--disable-filters --enable-filter=fps --enable-filter=framerate \
--enable-filter=fsync --enable-filter=gblur --enable-bsfs \
--disable-protocols --enable-protocol=file --enable-protocol=http --enable-protocol=https \
--disable-parsers --enable-parser=h264 --enable-parser=hevc \
--enable-parser=mpeg4video --enable-parser=mjpeg --enable-parser=png \
--disable-indevs --enable-indev=gdigrab --enable-indev=vfwcap --enable-indev=dshow \
--disable-outdevs \
--disable-libvpl --enable-hardcoded-tables \
--enable-hwaccel=h264_nvdec --enable-hwaccel=h264_dxva2 \
--enable-hwaccel=hevc_nvdec --enable-hwaccel=hevc_dxva2
```

FFmpeg 7.1，编译成静态库，UCRT64 环境，LGPL 2.1，MSVC 工具链：

```bash
./configure --enable-static --pkg-config-flags=--static \
--arch=x86_64 --toolchain=msvc --host-os=win64 --target-os=win64 --disable-debug \
--extra-cflags=-I/ucrt64/include --extra-ldflags=-L/ucrt64/lib \
--prefix=./build_msvc_static --enable-asm --enable-inline-asm \
--disable-doc --disable-htmlpages --disable-manpages --disable-podpages --disable-txtpages \
--enable-ffmpeg --disable-ffplay --disable-ffprobe \
--enable-avfilter --enable-avdevice --disable-swscale --disable-iconv \
--disable-decoders --enable-decoder=h264 --enable-decoder=hevc \
--enable-decoder=mpeg4 --enable-decoder=mjpeg --enable-decoder=aac \
--disable-encoders --enable-encoder=h264_nvenc --enable-encoder=hevc_nvenc \
--enable-encoder=mpeg4 --enable-encoder=mjpeg --enable-encoder=aac --enable-encoder=png \
--disable-demuxers --enable-demuxer=h264 --enable-demuxer=hevc \
--enable-demuxer=mpegvideo --enable-demuxer=mjpeg --enable-demuxer=aac \
--enable-demuxer=avi --enable-demuxer=mov --enable-demuxer=mpegps \
--disable-muxers --enable-muxer=h264 --enable-muxer=hevc \
--enable-muxer=mp4 --enable-muxer=mjpeg \
--enable-muxer=avi --enable-muxer=adts \
--disable-filters --enable-filter=fps --enable-filter=framerate \
--enable-filter=fsync --enable-filter=gblur --enable-bsfs \
--disable-protocols --enable-protocol=file --enable-protocol=http --enable-protocol=https \
--disable-parsers --enable-parser=h264 --enable-parser=hevc \
--enable-parser=mpeg4video --enable-parser=mjpeg --enable-parser=png \
--disable-indevs --enable-indev=gdigrab --enable-indev=vfwcap --enable-indev=dshow \
--disable-outdevs \
--enable-libvpl --enable-hardcoded-tables \
--enable-hwaccel=h264_nvdec --enable-hwaccel=h264_dxva2 \
--enable-hwaccel=hevc_nvdec --enable-hwaccel=hevc_dxva2
```

## x. FFmpeg 许可和法律注意事项

法律问题始终是问题和困惑的根源。这是试图澄清最重要的问题。通常的免责声明适用，这不是法律建议

### x.1 FFmpeg 许可证

FFmpeg 根据 [GNU 宽通用公共许可证 (LGPL) 2.1](http://www.gnu.org/licenses/old-licenses/lgpl-2.1.html) 版或更高版本获得许可。但是，[FFmpeg 合并了GNU 通用公共许可证 (GPL) 版本 2](http://www.gnu.org/licenses/old-licenses/gpl-2.0.html) 或更高版本中涵盖的几个可选部分和优化 。如果使用这些部分，则 GPL 适用于所有 FFmpeg。

阅读许可证文本，了解这如何影响构建在 FFmpeg 之上或重用 FFmpeg 的程序。您可能还希望查看 [GPL 常见问题解答](http://www.gnu.org/licenses/gpl-faq.html)。

请注意，FFmpeg 不可在任何其他许可条款下使用，尤其是专有/商业许可条款，甚至不能作为付费交换。

### x.2 许可证合规性清单

以下是链接 FFmpeg 库时 LGPL 合规性的清单。这不是遵守许可证的唯一方法，但我们认为这是最简单的方法。还有一些项目与 LGPL 合规性并不真正相关，但无论如何都是好主意。

- 编译 FFmpeg 时不带“--enable-gpl”，且不带“--enable-nonfree”。
- 使用动态链接（在 Windows 上，这意味着链接到 dll）来链接 FFmpeg 库。
- 分发 FFmpeg 的源代码，无论您是否修改过它。
- 确保源代码与您正在分发的库二进制文件完全对应。
- 在 FFmpeg 源代码的根目录中运行命令“git diff >changes.diff”以创建仅包含更改的文件。
- 解释如何在添加到源代码根目录的文本文件中编译 FFmpeg，例如配置行。
- 使用 tarball 或 zip 文件来分发源代码。
- 将 FFmpeg 源代码托管在与您分发的二进制文件相同的网络服务器上。
- 添加 "`This software uses code of <a href=http://ffmpeg.org>FFmpeg</a> licensed under the <a href=http://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>LGPLv2.1</a> and its source can be downloaded <a href=link_to_your_sources>here</a>`" 到您网站上有应用程序下载链接的每个页面。
- 在程序“关于框”中提及“此软件使用 LGPLv2.1 下的 FFmpeg 项目中的库”。
- 在您的 EULA 中提及您的程序使用 LGPLv2.1 下的 FFmpeg。
- 如果您的 EULA 声明对该代码的所有权，您必须明确 提及您不拥有 FFmpeg，以及在哪里可以找到相关所有者。
- 从您的 EULA 中删除任何逆向工程禁令。
v对 EULA 的所有翻译应用相同的更改。
- 不要拼错 FFmpeg（两个大写 F 和小写“mpeg”）。
- 不要将 FFmpeg dll 重命名为一些混淆的名称，但添加后缀或前缀就可以了（将“avcodec.dll”重命- 名为“MyProgDec.dll”不行，但可以重命名为“avcodec-MyProg.dll”）。
- 再次检查您编译到 FFmpeg 中的任何 LGPL 外部库（例如 LAME）的所有项目。
- 确保您的程序未使用任何 GPL 库（特别是 libx264）。

### x.3 商标

FFmpeg 是 FFmpeg 项目创始人 Fabrice Bellard 的商标。

## y. 参考文章

- [Windows编译和使用ffmpeg](https://blog.csdn.net/sinat_38854292/article/details/123234643)

- [win10下编译ffmpeg和x264](https://zhuanlan.zhihu.com/p/540376835)

- [FFmpeg 许可和法律注意事项](https://ffmpeg.github.net.cn/legal.html)
