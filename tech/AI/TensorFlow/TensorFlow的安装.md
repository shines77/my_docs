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

本文安装时的 `tensorflow` 版本号是 `2.17.0` 。大概需要下载 400 多MB的文件，安装后大约占 2 G左右的硬盘空间。

安装过程中得到下列错误信息，可能不影响使用：

```
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
torch 2.4.1 requires fsspec, which is not installed.
```

使用过程中可能会显示如下 warnning ：

```
I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
```

添加一个系统或者当前用户的环境变量 TF_ENABLE_ONEDNN_OPTS 即可，值为 0 。

## 2. TensorFlow GPU版本

如果怕有版本冲突，最好把 `TensorFlow` 安装到一个独立的 `Conda` 虚拟环境中。

### 2.1 选择版本

首先要选择合适的 `python` 版本，新建一个 `Conda` 虚拟环境，例如：

```bash
conda create --name TensorFlow python=3.7
conda activate TensorFlow
```

搜索 `cnDNN` 的版本：

```bash
conda search cudnn
```

搜索 `CUDA` 的版本：

```bash
conda search cudatoolkit
````

### 2.1 确定 TensorFlow 的版本

打开 `Tensorflow` 的官网：[Build from source on Windows | TensorFlow (google.cn)](https://tensorflow.google.cn/install/source_windows)

然后查看 Windows 中 GPU 与 Tensorflow、CUDA、cuDNN、python 直接的版本对应关系。

### 2.2 Tensorflow 的安装

例如你选择的是：cudnn 为 7.6.5，CUDA 为 10.1.243，Tensorflow 为 2.2.0 。

```bash
conda install cudatoolkit=10.1.243 cudnn=7.6.5
conda install tensorflow-gpu==2.2.0
```

由于跟 python 的版本也是相关的，所以一开始 `Conda` 的环境也要安装好所需要的 python 版本。

### 2.3 验证

```bash
python
import tensorflow as tf
tf.test.is_built_with_cuda()
```

## 3. 参考文章

- [tensorflow安装cpu版本](https://blog.51cto.com/u_16213610/10716737)

- [Tensorflow安装教程(完美安装gpu版本的tensorflow)（Windows,Conda,cuda,cudnn版本对应）](https://blog.csdn.net/qq_51800276/article/details/136485310)
