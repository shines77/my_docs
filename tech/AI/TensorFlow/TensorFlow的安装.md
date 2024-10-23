# TensorFlow的安装

## 1. TensorFlow CPU版本

### 1.1 Conda 环境

如果 `Conda` 环境下，默认个镜像源下载比较慢可以更换为国内的清华镜像源，但这一步不是必须的。

```bash
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/

conda config --set show_channel_urls yes
```

如果添加了镜像源导致无法下载 `tensorflow` ，建议先删除掉镜像，再重新安装：

```bash
conda config --remove-key channels
```

### 1.2 安装 CPU 版本

```bash
pip install --ignore-installed --upgrade tensorflow
```

安装 CPU 版本不需要事先安装 `CUDA` 和 `cuDNN` (还需要选择对应的 CUDA 和 cuDNN 版本)。

本文安装时的 `tensorflow` 版本号是 `2.17.0` 。

## 2. TensorFlow GPU版本

## x. 参考文章

- [Tensorflow安装教程(完美安装gpu版本的tensorflow)（Windows,Conda,cuda,cudnn版本对应）](https://blog.csdn.net/qq_51800276/article/details/136485310)
