# FFmpeg的常用命令

## 1. 常用命令

### ffmpeg -encoders

列出所有支持的编码器，包括硬件加速编码器（如 NVENC、VAAPI、QSV 等）。

```bash
ffmpeg -encoders
```

例如：

```bash
Encoders:
 V..... = Video
 A..... = Audio
 S..... = Subtitle
 .F.... = Frame-level multithreading
 ..S... = Slice-level multithreading
 ...X.. = Codec is experimental
 ....B. = Supports draw_horiz_band
 .....D = Supports direct rendering method 1
 ------
 V..... libx264              libx264 H.264 / AVC / MPEG-4 AVC / MPEG-4 part 10 (codec h264)
 V..... libx264rgb           libx264 H.264 / AVC / MPEG-4 AVC / MPEG-4 part 10 RGB (codec h264)
 V..... h264_vaapi           H.264/AVC (VAAPI) (codec h264)
 V..... libx265              libx265 H.265 / HEVC (codec hevc)
 V..... hevc_vaapi           H.265/HEVC (VAAPI) (codec hevc)
 ```

### ffmpeg -decoders

列出所有支持的解码器，包括硬件加速解码器（如 NVENC、VAAPI、QSV 等）。

```bash
ffmpeg -decoders
```

例如：

```bash
Decoders:
 V..... = Video
 A..... = Audio
 S..... = Subtitle
 .F.... = Frame-level multithreading
 ..S... = Slice-level multithreading
 ...X.. = Codec is experimental
 ....B. = Supports draw_horiz_band
 .....D = Supports direct rendering method 1
 ------
 VFS..D h264                 H.264 / AVC / MPEG-4 AVC / MPEG-4 part 10
 V..... h264_cuvid           Nvidia CUVID H264 decoder (codec h264)
 VFS..D hap                  Vidvox Hap
 VFS..D hevc                 HEVC (High Efficiency Video Coding)
 V..... hevc_cuvid           Nvidia CUVID HEVC decoder (codec hevc)
 ```

### ffmpeg -hwaccels

列出所有支持的硬件加速编码器，包括硬件加速编码器（如 NVENC、VAAPI、QSV 等）。

```bash
ffmpeg -hwaccels
```

例如：

```bash
Hardware acceleration methods:
cuda
vaapi
dxva2
d3d11va
qsv
cuvid
```

### ffmpeg -h encoder=<encoder_name>

查看某个编码器的详细信息。

```bash
ffmpeg -h encoder=<encoder_name>
```

例如：

```bash
ffmpeg -h encoder=h264_nvenc

Encoder h264_nvenc [NVIDIA NVENC H.264 encoder]:
    General capabilities: delay hardware
    Threading capabilities: none
    Supported pixel formats: yuv420p nv12 p010le yuv444p yuv444p16le bgr0 rgb0 cuda d3d11
...
```

## x. 参考文章

- [DeepSeek 大模型 2.5](https://chat.deepseek.com)
