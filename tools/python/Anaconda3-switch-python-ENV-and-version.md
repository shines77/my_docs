# Anaconda3 切换不同的 python 环境（版本）

## 1. 简介

有时候，我们希望电脑中有两种 `python` 环境，一种是 `2.x` 版本，另一种是 `3.x` 版本。传统的方法，同时安装 `python 2.x` 和 `python 3.x` 在同一台机器上很麻烦，会出现各种问题。安装了 `Anaconda` 后，就很容易的解决这个问题了。

## 2. 安装环境

安装过程不再敖述，具体请看：[https://blog.csdn.net/wz947324/article/details/80205181](https://blog.csdn.net/wz947324/article/details/80205181)

至于安装 `Anaconda2` 还是 `Anaconda3`，都可以，反正后面还可以安装另一个版本的 `python` 。两者的区别是，`Anaconda2` 默认安装的是 `python 2.x` 版本，而 `Anaconda3` 默认安装的是 `python 3.x` 版本。

下面以安装 `Anaconda3` 为例。

下文所使用的命令需在 `Anaconda` 的控制台环境下，即 `Anaconda Prompt` 环境，提示符如下所示：

```
(C:\Anaconda3) C:\Users\shines77>
```

### 2.1 检查版本

首先，我们用如下命令查看一下当前 `python` 环境的版本号，命令如下：

```
python --version
```

结果如下：

```
Python 3.6.3 :: Anaconda, Inc.
```

注：使用 `python -V` 命令也是一样的效果。

说明当前的 `python` 环境的版本是 `python 3.6.3` 。

### 2.2 检查已安装的环境

安装好 `Anaconda3` 之后，可以通过如下命令查看当前已经装好的 `python` 环境：

列出当前安装的所有 `python` 环境：

```
conda info -e
```

例如，显示如下结果：

```
# conda environments:
#
tensorflow               C:\Anaconda3\envs\tensorflow
root                  *  C:\Anaconda3
```

如上所示，可以看到，我的 `Anaconda3` 里安装了两种 `python` 环境。

第一个是 `python 3.x` 版本，专门为 `tensorflow` 准备的独立环境；第二个是 `Anaconda3` 默认安装的版本，也是 `python 3.x` 版本。

并且可以看到，当前激活的环境是 `root` 。

### 2.3 安装 python 2.x

在已经有了 `python 3.x` 环境的情况下，我们还想安装一个 `python 2.x` 环境。

使用如下命令，创建一个名为 `python27` 的环境，`python` 版本是 `2.7.x` ：

```
conda create --name python27 python=2.7
```

（注：`conda` 会为我们自动寻找 `2.7.x` 中的最新版本）。

执行结果如下：

```
Fetching package metadata .............
Solving package specifications: .

Package plan for installation in environment C:\Anaconda3\envs\python27:

The following NEW packages will be INSTALLED:

    certifi:        2018.11.29-py27_0
    pip:            18.1-py27_0
    python:         2.7.15-hcb6e200_5
    setuptools:     40.6.3-py27_0
    sqlite:         3.26.0-h0c8e037_0
    vc:             9-h7299396_1
    vs2008_runtime: 9.00.30729.1-hfaea7d5_1
    wheel:          0.32.3-py27_0
    wincertstore:   0.2-py27hf04cefb_0

Proceed ([y]/n)? y

vs2008_runtime 100% |###############################| Time: 0:00:01 659.26 kB/s
vc-9-h7299396_ 100% |###############################| Time: 0:00:00   1.46 MB/s
sqlite-3.26.0- 100% |###############################| Time: 0:00:00   3.30 MB/s
python-2.7.15- 100% |###############################| Time: 0:00:03   5.95 MB/s
certifi-2018.1 100% |###############################| Time: 0:00:00  10.21 MB/s
wincertstore-0 100% |###############################| Time: 0:00:00  13.93 MB/s
setuptools-40. 100% |###############################| Time: 0:00:00  10.56 MB/s
wheel-0.32.3-p 100% |###############################| Time: 0:00:00  13.75 MB/s
pip-18.1-py27_ 100% |###############################| Time: 0:00:00   9.69 MB/s
#
# （后面省略......）
#
```

## 3. 管理 python 环境（版本）

### 3.1 切换 python 环境

`python27` 环境安装完成以后，我们将把环境切换到 `python27`，命令如下：


```
activate python27
```

切换成功以后，可以看到提示符也改变了，例如：

```
(python27) C:\Users\shines77>
```

此时，可以使用第 `2.1` 和 `2.2` 小节的命令检查当前环境的版本和环境名称；

```
python --version

Python 2.7.15 :: Anaconda, Inc.
```

```
conda info -e

# conda environments:
#
python27              *  C:\Anaconda3\envs\python27
tensorflow               C:\Anaconda3\envs\tensorflow
root                     C:\Anaconda3
```

可以看到，当前 `python` 版本是 `Python 2.7.15`，激活的环境名称是 `python27` 。

### 3.2 回退 python 环境

退回到上一个 `python` 环境：

```
deactivate python27
```

### 3.2 删除 python 环境

删除一个 `python` 环境，命令如下：

```
conda remove --name python27 --all
```

## 4. 其他相关

### 4.1 `python2` 设置 utf-8

`python2` 的默认编码是 `ASCII`，修改为 `utf-8` 的方法：在 `Anaconda\Lib\site-packages` 目录下添加一个名字为 `sitecustomize.py` 文件，文件内容：

```
import sys  
sys.setdefaultencoding('utf-8')
```

### 4.2 设置国内源

在更新包的时候，默认源速度较慢，可以使用国内源：

```
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --set show_channel_urls yes
```

### 4.3 安装指定版本的 TensorFlow


1. 首先在 `Anaconda` 的库中查找所有的相关的 `repository`：

```
anaconda search -t conda tensorflow
```
 
2. 根据自己的环境选择安装对应的版本，查看 `repository` 中的信息，`Anaconda` 会返回供安装的版本号以及安装方法：

```
anaconda show anaconda/tensorflow
```
 
3. 根据返回的内容进行安装：

```
# 在 Linux 上面亲测通过，Windowns 下面未找到包

conda install --channel https://conda.anaconda.org/anaconda tensorflow=1.6.0
```

## 5. 参考文章

1. [`Anaconda3 不同版本 python 环境的安装及切换`](https://blog.csdn.net/wz947324/article/details/80228679)

    [https://blog.csdn.net/wz947324/article/details/80228679](https://blog.csdn.net/wz947324/article/details/80228679)

2. [`Anaconda3 多版本 Python 管理以及 TensorFlow 版本的选择安装`](https://www.cnblogs.com/wxshi/p/6805120.html)

    [https://www.cnblogs.com/wxshi/p/6805120.html](https://www.cnblogs.com/wxshi/p/6805120.html)