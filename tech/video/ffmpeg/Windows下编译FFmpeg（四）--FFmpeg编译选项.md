# Windows下编译FFmpeg（四）-- FFmpeg编译选项

## 导航

- [Windows下编译FFmpeg（一）-- MSYS2和mingw-w64](./Windows下编译FFmpeg（一）--MSYS2和mingw-w64.md)
- [Windows下编译FFmpeg（二）-- MinGW32和msys 1.0](./WWindows下编译FFmpeg（二）--MinGW32和msys-1.0.md)
- [Windows下编译FFmpeg（三）-- 其他依赖库的下载](./Windows下编译FFmpeg（三）--其他依赖库的下载.md)
- [Windows下编译FFmpeg（四）-- FFmpeg编译选项](./Windows下编译FFmpeg（四）--FFmpeg编译选项.md)

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
