
# Ubuntu 16.04 更新 CMake 版本到最新版

## 1. 更新到最新版

安装流程如下:

```bash
cd /root/downloads
wget https://cmake.org/files/v3.18/cmake-3.18.0-rc4-Linux-x86_64.tar.gz
tar zxvf cmake-3.18.0-rc4-Linux-x86_64.tar.gz

sudo mv cmake-3.18.0-rc4-Linux-x86_64  /opt/cmake-3.18
sudo ln -sf /opt/cmake-3.18/bin/*  /usr/bin/
```

## 2. 参考文章:

* [Ubuntu 16.04 安装 CMake 3.11 及简单使用](https://blog.csdn.net/u012101561/article/details/80413627)
