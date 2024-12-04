# 关于 FFmpeg 的 av_register_all() 函数被废弃

## 1. 原因

从 FFmpeg 4.0 开始，av_register_all() 函数已被废弃，并且在未来的版本中可能会被移除。av_register_all() 函数在旧版本的 FFmpeg 中用于注册所有的复用器（muxer）、解复用器（demuxer）、编码器（encoder）、解码器（decoder）和协议等。

解释：

FFmpeg 团队为了减少库的依赖性和体积，将注册函数移到了特定的库中。如果你只使用某些编解码器或者协议，你可以只链接那些你需要的库，从而减少程序的大小和对系统资源的需求。

更确切的废弃信息来自于 APIchanges 中：

```text
2018-02-06 - 0694d87024 - lavf 58.9.100 - avformat.h
  Deprecate use of av_register_input_format(), av_register_output_format(),
  av_register_all(), av_iformat_next(), av_oformat_next().
  Add av_demuxer_iterate(), and av_muxer_iterate().
```

也就是 2018-02-06，从 avformat 库的 58.9.100 版本开始，大约是 FFmpeg 的 4.0 版本前后。

## 2. 解决方法

如果你正在编写一个新的应用程序，你应该直接链接你需要的库，而不是使用 av_register_all()。

例如，如果你只需要使用 libavcodec 中的 h264 解码器，你应该只链接 libavcodec 和 libavformat。

对于旧代码，你需要根据你的需求，手动替换掉 av_register_all()，这通常意味着你需要显式地注册你需要的解码器和协议。

例如：

- 你可以使用 avcodec_register_all() 来加载所有编解码器。

- 你可以使用 avformat_register_all() 来加载所有文件格式。

- 你可以使用 av_protocol_register_all() 来加载所有协议。

确保你的代码只链接必要的库，以减少程序的大小和潜在的安全风险。

示例代码：

```c
// 旧的使用 av_register_all() 的方式
av_register_all();

// 新的显式注册方式
avcodec_register_all();
avformat_register_all();
av_protocol_register_all();

// 或者只注册特定的编解码器
avcodec_register(&h264_decoder);

// 或者只注册特定的文件格式和协议
avformat_network_init();
av_protocol_register(&file_protocol);
```

如果加上预编译开关，可以这样写：

```c
    // 初始化FFmpeg
#if LIBAVFORMAT_BUILD < AV_VERSION_INT(58, 9, 100)
    // 该函数从 avformat 库的 58.9.100 版本开始被废弃
    // (2018-02-06, 大约是 FFmepg 4.0 版本)
    av_register_all();
#endif
```

## 3. 参考文章

- [百度AI智能回答](https://chat.baidu.com/)
