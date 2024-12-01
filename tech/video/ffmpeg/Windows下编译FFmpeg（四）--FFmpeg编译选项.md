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

## x. FFmpeg 许可和法律注意事项

法律问题始终是问题和困惑的根源。这是试图澄清最重要的问题。通常的免责声明适用，这不是法律建议

### x.1 FFmpeg 许可证

FFmpeg 根据 [GNU 宽通用公共许可证 (LGPL) 2.1](http://www.gnu.org/licenses/old-licenses/lgpl-2.1.html) 版或更高版本获得许可。但是，[FFmpeg 合并了GNU 通用公共许可证 (GPL) 版本 2](http://www.gnu.org/licenses/old-licenses/gpl-2.0.html) 或更高版本中涵盖的几个可选部分和优化 。如果使用这些部分，则 GPL 适用于所有 FFmpeg。

阅读许可证文本，了解这如何影响构建在 FFmpeg 之上或重用 FFmpeg 的程序。您可能还希望查看 [GPL 常见问题解答](http://www.gnu.org/licenses/gpl-faq.html)。

请注意，FFmpeg 不可在任何其他许可条款下使用，尤其是专有/商业许可条款，甚至不能作为付费交换。

### x.2 许可证合规性清单

以下是链接 FFmpeg 库时 LGPL 合规性的清单。这不是遵守许可证的唯一方法，但我们认为这是最简单的方法。还有一些项目与 LGPL 合规性并不真正相关，但无论如何都是好主意。

- 编译 FFmpeg 时不带“--enable-gpl”且不 带“--enable-nonfree”。
- 使用动态链接（在 Windows 上，这意味着链接到 dll）来链接 FFmpeg 库。
- 分发 FFmpeg 的源代码，无论您是否修改过它。
- 确保源代码与您正在分发的库二进制文件完全对应。
- 在 FFmpeg 源代码的根目录中运行命令“git diff >changes.diff”以创建仅包含更改的文件。
- 解释如何在添加到源代码根目录的文本文件中编译 FFmpeg，例如配置行。
- 使用 tarball 或 zip 文件来分发源代码。
- 将 FFmpeg 源代码托管在与您分发的二进制文件相同的网络服务器上。
- 添加“此软件使用根据 <a href=http://www.gnu.org/licenses/old-licenses/lgpl-2.1 许可的 <a href=http://ffmpeg.org>FFmpeg</a> 代码.html>LGPLv2.1</a> 及其源代码可以在<a href=link_to_your_sources>此处</a>下载到您网站上有应用程序下载链接的每个页面。
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

- [FFmpeg 许可和法律注意事项](https://ffmpeg.github.net.cn/legal.html)
