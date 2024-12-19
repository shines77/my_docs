# 关于 dshow 中的 qedit.h 头文件缺失的问题

## 1. 前言

Windows 7 SDK 中带了 DirectShow 的头文件和 strmiids.lib，quartz.lib，也带了 DirectX 的 Samples，`strmbase.lib` 可以自己用 Samples 里的 `Microsoft SDKs\Windows\v7.1\Samples\multimedia\directshow\baseclasses` 工程编译。

但是没有 `qedit.h` ，需要安装 DirectX 9.0b SDK 完整包或者 DirectX 9.0 SDK Extras 。`qedit.h` 有 `ISampleGrabber` 等接口的定义，用 DShow 对摄像头和麦克风抓取数据的时候要用到。

DirectX 9.0b SDK 现在已经很难下载到了，CSDN 上有下载，但是都是要会员才能下载。可以百度、bing、google 搜索一下，这里有一个可能能下载的地址：

```bash
https://www.fileplanet.com/archive/p-16004/DirectX-9-0-Software-Development-Kit-with-DirectX-9-0b-Runtime
```

## 2. qedit.h 问题

下载完成后，不用安装，其实它是一个压缩文件，用 7-zip 解压到目录，它嵌套了两层压缩，解压两次即可。你不用安装它，如果你是 Windows 7, 10, 11，你也安装不了，因为它只支持 Win 7 以前的 32 位系统。

直接拷贝其目录下的 \Include\qedit.h 文件到你当前 MSVC 的 include 文件夹即可，但是要做以下修改。

打开 `qedit.h` ，把第 491 行的 "dxtrans.h" 注释掉。因为使用了它，后续还要引入很多没必要的文件。

```cpp
/* header files for imported files */
#include "oaidl.h"
#include "ocidl.h"
//#include "dxtrans.h"   -- Line 491
#include "amstream.h"
```

然后，在你的代码中，在 #include <qedit.h> 之前定义如下宏，例如：

```cpp
#define __IDxtCompositor_INTERFACE_DEFINED__
#define __IDxtAlphaSetter_INTERFACE_DEFINED__
#define __IDxtJpeg_INTERFACE_DEFINED__
#define __IDxtKey_INTERFACE_DEFINED__

#include <qedit.h>      // For ISampleGrabber
```

这样就完美的避开了对 "dxtrans.h" 文件的依赖。如果编译过程中，仍然提示 XX 头文件找不到，去 DirectX 9.0b SDK 解压目录下的 \Include 拷贝同名的文件到你当前 MSVC 的 include 文件夹即可，应该不会很多。如果引用 "dxtrans.h" 文件的话，需要拷贝的文件很多，而且有很多错误。

搞定！完美解决。

## 3. 参考文章

- [Qedit.h问题](https://blog.csdn.net/jsmiler/article/details/6495763)

- [提示缺少Qedit.h问题](https://blog.csdn.net/joeblackzqq/article/details/10944005)
