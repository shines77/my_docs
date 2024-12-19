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

### 1:1 HWACCEL Transcode without Scaling

以下命令读取 input.mp4 文件，并将其转码为具有相同分辨率和相同音频编解码器的 H.264 视频格式的 output.mp4 。

```bash
ffmpeg -y -vsync 0 -hwaccel cuda -hwaccel_output_format cuda -i input.mp4 -c:a copy -c:v h264_nvenc -b:v 5M output.mp4
```

### 1:1 HWACCEL Transcode with Scaling

以下命令读取 input.mp4 文件，并使用 720p 分辨率的 H.264 视频和相同的音频编解码器将其转码为 output.mp4 。以下命令使用 cuvid 解码器中的内置 resizer 。

```bash
ffmpeg -y -vsync 0 -hwaccel cuda -hwaccel_output_format cuda –resize 1280x720 -i input.mp4 -c:a copy -c:v h264_nvenc -b:v 5M output.mp4
```

### 1:N HWACCEL encode from YUV or RAW Data

从 YUV 或 RAW 文件进行编码可能会导致磁盘 I/O 成为瓶颈，建议从 SSD 进行此类编码以获得最佳性能。以下命令读取 input.yuv 文件，并以不同的输出比特率将其编码为四个不同的 H.264 视频。请注意，此命令仅对所有编码操作加载一个 YUV，从而提高了磁盘 I/O 的效率，提高了整体编码性能。

Input: input.yuv (420p, 1080p)

Outputs: 1080p (8M), 1080p (10M), 1080p (12M), 1080p (14M)

```bash
ffmpeg -y -vsync 0 -pix_fmt yuv420p -s 1920x1080 -i input.yuv -filter_complex "[0:v]hwupload_cuda,split=4[o1][o2][o3][o4]" -map "[o1]" -c:v h264_nvenc -b:v 8M output1.mp4 -map "[o2]" -c:v h264_nvenc -b:v 10M output2.mp4 -map "[o3]" -c:v h264_nvenc -b:v 12M output3.mp4 -map "[o4]" -c:v h264_nvenc -b:v 14M output4.mp4
```

### Video Encoding

编码视频的质量取决于编码器使用的各种功能。要对 720p YUV 进行编码，请使用以下命令。

```bash
ffmpeg -y -vsync 0 –s 1280x720 –i input.yuv -c:v h264_nvenc output.mp4
```

这将生成具有H264编码视频的MP4格式的输出文件（output.MP4）。

视频编码可以大致分为两类用例：

- **延迟容忍的高质量**：在这种用例中，延迟是允许的。可以使用编码器功能，如 B 帧、前瞻、参考 B 帧、可变比特率（VBR）和更高的 VBV 缓冲区大小。典型用例包括云转码、录制和归档等。

- **低延迟**：在这种用例中，延迟应该很低，可以低至 16 毫秒。在这种模式下，B 帧被禁用，使用恒定比特率模式，VBV 缓冲区大小保持很低。典型的用例包括实时游戏、直播和视频会议等。由于上述约束，这种编码模式导致较低的编码质量。

NVENCODE API 支持通过 FFmpeg 命令行显示的用于调整质量、性能和延迟的几个功能。建议根据用例启用功能和命令行选项。

### Video Decoding

FFmpeg 视频解码器使用起来很简单。要解码 input.mp4 中的输入比特流，请使用以下命令。

```bash
ffmpeg -y -vsync 0 -c:v h264_cuvid -i input.mp4 output.yuv
```

这将生成 NV12 格式的输出文件（output.yuv）。

### Low Latency High Quality

Input: input.mp4 (30fps)

Output: same resolution as input, bitrate = 5M (audio same as input)

```bash
ffmpeg -y -vsync 0 -hwaccel cuda -hwaccel_output_format cuda -i input.mp4 -c:a copy -c:v h264_nvenc -preset p6 -tune ll -b:v 5M -bufsize 167K -maxrate 10M -qmin 0 output.mp4
```

### Low Latency High performance

Use -preset p2 instead of -preset p6 in above command line.

```bash
ffmpeg -y -vsync 0 -hwaccel cuda -hwaccel_output_format cuda -i input.mp4 -c:a copy -c:v h264_nvenc -preset p2 -tune ll -b:v 5M -bufsize 167K -maxrate 10M -qmin 0 output.mp4
```

### 列出所有 dshow 支持的视频，音频设备

```bash
ffmpeg -list_devices true -f dshow -i dummy
```

### Windows 音视频录制

录制摄像头和麦克风的音视频，保存为 H.264 和 AAC 格式，视频码率为 2000kb /s，音频码率为 128 kb/s，图像格式为默认 (yuv422p)，图像大小为摄像头默认大小 (640x480)：

```bash
.\ffmpeg -f dshow -i video="HD WebCam" -f dshow -i audio="麦克风 (Realtek High Definition Audio)" -vcodec libx264 -b:v 2000k -acodec aac -b:a 128k -strict -2 mycamera.mp4
```

跟以上相同，图像格式改为 yuv420p ，压缩率更高一些：

```bash
.\ffmpeg -f dshow -i video="HD WebCam" -f dshow -i audio="麦克风 (Realtek High Definition Audio)" -vcodec libx264 -b:v 2000k -pix_fmt yuv420p -acodec aac -b:a 128k -strict -2 mycamera.mp4
```

## x. 参考文章

- [DeepSeek 大模型 2.5](https://chat.deepseek.com)

- [Using FFmpeg with NVIDIA GPU Hardware Acceleration](https://docs.nvidia.com/video-technologies/video-codec-sdk/12.2/ffmpeg-with-nvidia-gpu/index.html)
