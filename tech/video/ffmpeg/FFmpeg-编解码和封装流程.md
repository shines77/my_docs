# FFmpeg 编码、解码和封装流程

## 1. av_register_all()

av_register_all() 函数在旧版本的 FFmpeg 中用于注册所有可用的音视频复用器（muxer）、封装器（demuxer）、编码器（encoder）、解码器（decoder）和协议等，以便在处理音视频文件时可以正确地识别和处理各种格式。

从 FFmpeg 4.0 或更高版本中，这个函数已经被废弃，取而代之的是自动注册机制。现在，当你使用 FFmpeg 库时，它会自动注册可用的封装器和解码器，而无需显式调用 av_register_all()。

因此，在使用较新版本的 FFmpeg 时，你无需调用 av_register_all() 函数。FFmpeg 团队为了减少库的依赖性和体积，将注册函数移到了特定的库中。如果你只使用某些编解码器或者协议，你可以只链接那些你需要的库，从而减少程序的大小和对系统资源的需求。

例如：

- 你可以使用 avcodec_register_all() 来加载所有编解码器。

- 你可以使用 avformat_register_all() 来加载所有文件格式。

- 你可以使用 av_protocol_register_all() 来加载所有协议。

更详细的内容，请看另一篇文章：[关于 FFmpeg 的 av_register_all() 函数被废弃](./关于FFmpeg的av_register_all()函数被废弃.md) 。

## 2. avformat_network_init()

avformat_network_init() 用于初始化网络协议。在某些情况下，当你需要通过网络访问音视频资源时，你可能需要调用这个函数来确保网络协议的正确初始化。

该函数位于 libavformat/network.h 头文件中。调用 avformat_network_init() 函数会初始化 FFmpeg 库中使用的网络协议，以便可以使用诸如 HTTP、RTMP、RTSP 等网络协议来访问远程音视频资源。

以下是示例代码，展示了如何使用 avformat_network_init() 函数：

```c
#include <libavformat/avformat.h>
int main()
{
    // 初始化网络协议
    avformat_network_init();

    // 其他音视频处理操作
    ...

    // 清理资源
    avformat_network_deinit();
    return 0;
}
```

在上述示例中，我们首先调用 avformat_network_init() 来初始化网络协议。然后，在这个函数之后，你可以进行其他音视频处理操作。最后，使用 avformat_network_deinit() 清理相关资源。

## 3. avformat_open_input()

avformat_open_input() 用于打开音视频文件或网络流以进行读取操作。

函数原型：

```c
int avformat_open_input(AVFormatContext **ps, const char *url,
                        AVInputFormat *fmt, AVDictionary **options);
```

参数说明：

- **ps**：指向 AVFormatContext 指针的指针。函数将分配并填充一个 AVFormatContext 结构体，用于表示打开的音视频文件或网络流的上下文信息。

- **url**：要打开的音视频文件或网络流的路径或 URL。

- **fmt**：可选参数，用于指定强制使用的输入格式。通常可以将其设置为 NULL，让 FFmpeg 自动检测输入格式。

- **options**：可选参数，用于传递额外的选项。可以使用 AVDictionary 结构体来设置各种选项，比如设置输入缓冲区大小等。

返回值：

如果返回值大于等于 0，表示函数的执行结果。如果返回值小于 0，则表示打开输入失败，可以通过返回值查看错误码。

示例代码：

```c
#include <libavformat/avformat.h>

int main()
{
    AVFormatContext *formatContext = NULL;

    // 打开音视频文件或网络流
    int ret = avformat_open_input(&formatContext, "test01.mp4", NULL, NULL);
    if (ret < 0) {
        // 打开失败，处理错误
        return ret;
    }

    // 在这里进行其他音视频处理操作

    // 关闭输入文件或网络流
    avformat_close_input(&formatContext);
    return 0;
}
```

在上述示例中，我们首先声明了一个 AVFormatContext 指针 formatContext。

然后，调用 avformat_open_input() 函数来打开名为 test01.mp4 的音视频文件。如果打开成功，将返回一个非负值；否则，将返回一个负数，表示打开失败。在这之后，你可以进行其他的音视频处理操作。

最后，使用 avformat_close_input() 函数关闭输入文件或网络流，并释放相关资源。

> 这只是 avformat_open_input() 函数的基本用法。在实际使用中，你可能需要根据需要设置更多的选项、处理错误以及对音视频流进行解码等操作。

## 4. avformat_find_stream_info()

avformat_find_stream_info() 用于获取音视频流的相关信息。

函数原型：

```c
int avformat_find_stream_info(AVFormatContext *ic, AVDictionary **options);
```

参数说明：

- **ic**：指向 AVFormatContext 结构体的指针，表示打开的音视频文件或网络流的上下文信息。

- **options**：可选参数，用于传递额外的选项。可以使用 AVDictionary 结构体来设置各种选项。

返回值：

如果返回值大于等于 0，表示函数的执行结果。如果返回值小于 0，则表示获取流信息失败，可以通过返回值查看错误码。

示例代码：

```c
#include <libavformat/avformat.h>

int main()
{
    AVFormatContext *formatContext = NULL;

    // 打开音视频文件或网络流
    int ret = avformat_open_input(&formatContext, "test01.mp4", NULL, NULL);
    if (ret < 0) {
        // 打开失败，处理错误
        return ret;
    }

    // 获取音视频流的相关信息
    ret = avformat_find_stream_info(formatContext, NULL);
    if (ret < 0) {
        // 获取信息失败，处理错误
        avformat_close_input(&formatContext);
        return ret;
    }

    // 在这里可以访问音视频流的相关信息

    // 关闭输入文件或网络流
    avformat_close_input(&formatContext);
    return 0;
}
```

在上述示例中，我们首先调用 avformat_open_input() 函数来打开名为 test01.mp4 的音视频文件。如果打开成功，将返回一个非负值；否则，将返回一个负数，表示打开失败。

然后，我们调用 avformat_find_stream_info() 函数来获取音视频流的相关信息。如果获取成功，将返回一个非负值；否则，将返回一个负数，表示获取信息失败。在获取成功后，你可以访问 AVFormatContext 结构体中的成员来获取音视频流的详细信息，比如流的数量、流的类型、时长等。

最后，使用 avformat_close_input() 函数关闭输入文件或网络流，并释放相关资源。

> 注意: 获取流信息是一个耗时的操作，可能需要一些时间。在实际使用中，你可能还需要对流进行解码、选择特定的音视频流等操作。

## 5. av_find_best_stream()

av_find_best_stream() 用于查找最佳的音视频流。

函数原型：

```c
int av_find_best_stream(AVFormatContext *ic, enum AVMediaType type,
                        int wanted_stream_nb, int related_stream,
                        AVCodec **decoder_ret, int flags);
```

参数说明：

- **ic**：指向 AVFormatContext 结构体的指针，表示打开的音视频文件或网络流的上下文信息。

- **type**：希望获取的流的类型，可以是 AVMEDIA_TYPE_AUDIO、AVMEDIA_TYPE_VIDEO 或 AVMEDIA_TYPE_SUBTITLE。

- **wanted_stream_nb**：期望获取的流的索引号。如果为负值，则表示不关心具体的索引号，只需获取符合类型要求的最佳流。

- **related_stream**：关联流的索引号。在某些情况下，需要提供关联的流索引号来帮助确定最佳流。

- **decoder_ret**：指向 AVCodec 指针的指针，用于返回找到的解码器。

- **flags**：附加选项，可以设置为 AVFMT_FIND_STREAM_INFO_DISCARD, AVFMT_FIND_STREAM_INFO_NOBLOCK 或 AVFMT_FIND_STREAM_INFO_NOBSF。

返回值：

如果返回值大于等于 0，表示找到的最佳流的索引号。如果返回值小于 0，则表示未找到符合要求的流。

示例代码：

```c
#include <libavformat/avformat.h>

int main()
{
    AVFormatContext *formatContext = NULL;
    int audioStreamIndex = -1;
    int videoStreamIndex = -1;

    // 打开音视频文件或网络流
    int ret = avformat_open_input(&formatContext, "test01.mp4", NULL, NULL);
    if (ret < 0) {
        // 打开失败，处理错误
        return ret;
    }

    // 获取音频流和视频流的索引号
    audioStreamIndex = av_find_best_stream(formatContext, AVMEDIA_TYPE_AUDIO, -1, -1, NULL, 0);
    videoStreamIndex = av_find_best_stream(formatContext, AVMEDIA_TYPE_VIDEO, -1, -1, NULL, 0);

    // 在这里可以使用音频流和视频流进行处理

    // 关闭输入文件或网络流
    avformat_close_input(&formatContext);
    return 0;
}
```

在上述示例中，我们首先调用 avformat_open_input() 函数来打开名为 test01.mp4 的音视频文件。

然后，我们使用 av_find_best_stream() 函数分别查找最佳的音频流和视频流。通过指定 AVMEDIA_TYPE_AUDIO 和 AVMEDIA_TYPE_VIDEO，我们可以获取符合类型要求的最佳流的索引号。在这之后，你可以使用获取到的音频流和视频流进行相应的处理。

最后，使用 avformat_close_input() 函数关闭输入文件或网络流，并释放相关资源。

> av_find_best_stream() 函数会根据流的相关属性（例如编解码器参数、时长等）来选择最佳的流。你可以根据需要对返回的流进行进一步的处理，例如解码等操作。

## x. 参考文章

- [深入理解ffmpeg解封装流程](https://zhuanlan.zhihu.com/p/677977011)
