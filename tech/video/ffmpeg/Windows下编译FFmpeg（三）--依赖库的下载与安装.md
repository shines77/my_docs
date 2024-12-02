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

SDL2 依赖：

```bash
pacman -S mingw-w64-ucrt-x86_64-SDL2
```

INTEL 的 CPU 里面也有一套硬件视频编解码器，它用到的库叫做 libvpl。我们先安装 libvpl：

```bash
pacman -S ucrt64/mingw-w64-ucrt-x86_64-libvpl
```

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

## x. 参考文章

- [在windows上编译FFmpeg](https://zhuanlan.zhihu.com/p/707298876)

- []()
