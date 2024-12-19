# FFmpeg 编解码知识

## 1. FFmpeg 组件

FFmpeg 是库和工具的集合来处理多媒体内容，如音频、视频、字幕和相关的元数据。

包含以下库：

```cpp
libavcodec      // 提供了更广泛的编解码器的实现。
libavformat     // 实现流协议，容器格式和基本I / O访问。
libavutil       // 包括哈希尔，解压缩器和杂项效用函数。
libavfilter     // 提供了通过一系列过滤器来改变已解码的音频和视频的意思。
libavdevice     // 提供了访问捕获和播放设备的抽象。
libswresample   // 实现音频混合和重采样程序。
libswscale      // 实现颜色转换和缩放程序。
```

工具：

```cpp
ffmpeg          // 是用于操纵，转换和流式传输多媒体内容的命令行工具箱。
ffplay          // 是一个简约的多媒体播放器。
ffprobe         // 是一种检查多媒体内容的简单分析工具。
ffserver        // 是一种多媒体流媒体服务器，用于直播。
```

## 2. FFmpeg 解码函数

FFmpeg 解码函数简介：

```cpp
av_register_all()               // 注册所有组件，此函数 4.0 以上版本已经废弃。
avformat_open_input()           // 打开输入视频文件。
avformat_find_stream_info()     // 获取视频文件信息。
avcodec_find_decoder()          // 查找解码器。
avcodec_open2()                 // 打开解码器。
av_read_frame()                 // 从输入文件读取一帧压缩数据。
avcodec_decode_video2()         // 解码一帧压缩数据，此函数后续版本已经废弃。
avcodec_close()                 // 关闭解码器。
avformat_close_input()          // 关闭输入视频文件。
```

使用avformat_open_input() 函数可以打开一个视频文件，获取时网络摄像头的rtsp地址。详解这里就不多说了，可以看雷神的该函数的解析。

## 3. FFmpeg 数据结构

**AVFormatContext**

封装格式上下文结构体，也是统领全局的结构体，保存了视频文件封装格式相关信息。

```cpp
iformat：    输入视频的 AVInputFormat
nb_streams： 输入视频的 AVStream 个数
streams：    输入视频的 AVStream []数组
duration：   输入视频的时长（以微秒为单位）
bit_rate：   输入视频的码率
```

**AVInputFormat**

每种封装格式（例如 FLV, MKV, MP4, AVI）对应一个该结构体。

```cpp
name：       封装格式名称
long_name：  封装格式的长名称
extensions： 封装格式的扩展名
id：         封装格式ID
// 一些封装格式处理的接口函数
```

**AVStream**

视频文件中每个视频（音频）流对应一个该结构体。

```cpp
id：序号
codec：          流对应的AVCodecContext
time_base：      该流的时基
r_frame_rate：   该流的帧率
```

**AVCodecContext**

编码器上下文结构体，保存了视频（音频）编解码相关信息。

```cpp
codec：      编解码器的AVCodec
width：      图像的宽（只针对视频）
height:      图像的高（只针对视频）
pix_fmt：    像素格式（只针对视频）
sample_rate：采样率（ 只针对音频）
channels：   声道数（只针对音频）
sample_fmt： 采样格式（只针对音频）
```

**AVCodec**

每种视频（音频）编解码器（例如 H.264 解码器）对应一个该结构体。

```cpp
name：       编解码器名称
long_name：  编解码器长名称
type：       编解码器类型
id：         编解码器ID
// 一些编解码的接口函数
```

**AVPacket**

存储一帧压缩编码数据。

```cpp
pts：        显示时间戳
dts：        解码时间戳
data：       压缩编码数据
size：       压缩编码数据大小
stream_index ：所属的 AVStream
```

**AVFrame**

存储一帧解码后像素（采样）数据。

```cpp
data：       解码后的图像像素数据（音频采样数据）。
linesize：   对视频来说是图像中一行像素的大小；对音频来说是整个音频帧的大小。
width：      图像的宽（只针对视频）。
height:     图像的高（只针对视频）。
key_frame：  是否为关键帧（只针对视频）。
pict_type：  帧类型（只针对视频）。例如：I， P， B。
```

## 4. 释放资源的顺序

```cpp
sws_freeContext(pSwsContext);
av_frame_free(&pAVFrame);
avcodec_close(pAVCodecContext);
avformat_close_input(&pAVFormatContext);
```

这个顺序不能错，如果想关闭一个摄像头的取流地址不能单独调用 avformat_close_input(&pAVFormatContext); 因为你释放掉这个内存，里面的一些结构体没有被释放会导致程序崩溃。


## 5. 参考文章

- [FFmpeg再学习 -- FFmpeg解码知识](https://blog.csdn.net/qq_29350001/article/details/75529620)

- [关于FFmpeg释放 AVFormatContext*解码上下文的一些问题](https://www.cnblogs.com/lidabo/p/17623739.html)
