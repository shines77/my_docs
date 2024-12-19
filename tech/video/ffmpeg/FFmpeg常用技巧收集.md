# FFmpeg 常用技巧收集

## 1. 视频

- [FFmpeg 开发之 AVFilter 使用流程总结](https://www.cnblogs.com/lidabo/p/15963533.html) | 推荐指数：★★★

## 2. 音频

- [ffplay源码分析6-音频重采样](https://www.cnblogs.com/leisure_chn/p/10312713.html) | 推荐指数：★★★★

## 3. 时间戳

- [FFmpeg时间戳详解](https://www.cnblogs.com/leisure_chn/p/10584910.html) | 推荐指数：★★★★★

- [ffplay源码分析4-音视频同步 ](https://www.cnblogs.com/leisure_chn/p/10307089.html) | 推荐指数：★★★★★

## 4. DirectShow


## 5. 录屏/摄像头

- [如何用FFmpeg API采集摄像头视频和麦克风音频，并实现录制文件的功能](https://www.cnblogs.com/lidabo/p/8662955.html) | 推荐指数：★★★★

- [FFmpeg获取DirectShow设备数据（摄像头，录屏）](https://blog.csdn.net/leixiaohua1020/article/details/38284961) | 推荐指数：★★★



## 6. 收流/推流

- [FFmpeg流媒体处理-收流与推流](https://www.cnblogs.com/leisure_chn/p/10623968.html) | 推荐指数：★★★★★

- [最简单的基于FFmpeg的推流器（以推送RTMP为例）](https://blog.csdn.net/leixiaohua1020/article/details/39803457) | 推荐指数：★★★

- [ffmpeg播放RTSP的一点优化](https://www.cnblogs.com/lidabo/p/17510822.html) | 推荐指数：★★★

    ```cpp
    // 1. 画质优化: 通过增大“buffer_size”参数来提高画质，减少花屏现象
    av_dict_set(&options, "buffer_size", "1024000", 0);

    // 2. 如设置 20s 超时，默认参数打开 RTSP 流时，若连接不上，会出现卡死在打开函数的情况
    av_dict_set(&options, "stimeout", "20000000", 0);

    // 3. 最大延迟时间
    av_dict_set(&options, "max_delay", "500000", 0);

    // 4. 以 tcp 方式打开，也可以选择 udp
    av_dict_set(&options, "rtsp_transport", "tcp", 0);
    ```


## 7. 直播

- [总结：从一个直播APP看流媒体系统的应用](https://mp.weixin.qq.com/s/G6zE4iokEfcZQHHrm2xe1w)


## 8. 播放器



## 9. 格式转换

- [FFmpeg编解码处理1-转码全流程简介](https://www.cnblogs.com/leisure_chn/p/10584901.html) | 推荐指数：★★★★



## 10. 流服务器


## 11. FFmpeg编译



## 12. 其他

### 1. 推流端框架：

- 采集：AVFoundation
- 滤镜：GPUImage
- 编码：FFmpeg/X264/Speex
- 推流：Librtmp

### 2. 流媒体服务器

- nginx-rtmp
- SRS
- BMS

### 3. 播放端

- 解码：FFmpeg/X264
- 播放：ijkplayer/video.js/flv.js
