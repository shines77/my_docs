
Ubuntu 14.04 安装 GCC 4.8.x, 4.9.x, 5.4 和 6.0 并随意切换
============================================================

## 1. 添加 PPA 源 ##

在 `toolchain/test` 下已经有打包好的 `gcc` ，版本有 `4.x`、`5.0`、`6.0` 等，用这个 `PPA` 升级 `gcc` 就可以啦！

首先添加 `ppa` 到更新源，并更新 `apt-get` ：

```shell
# 安装 add-apt-repository 组件
$ sudo apt-get install -y software-properties-common

# 添加 ppa 源
$ sudo add-apt-repository ppa:ubuntu-toolchain-r/test

# 更新 apt 源
$ sudo apt-get update
```

其中第二句执行的时候，如果你已经安装过 `add-apt-repository` 了的话，按回车确认，继续即可。

如果出现下面的错误信息或其他错误提示，则说明 `add-apt-repository` 没有安装或无法正常工作：

```shell
sudo: add-apt-repository: command not found
```

注意：在添加了 `ppa` 源以后，一定要记得执行 `sudo apt-get update` 命令，才会让添加的 `ppa` 生效。

## 2. 安装 gcc 各个版本 ##

`Ubuntu 14.04` 系统更新源默认安装的版本是 `gcc-4.8`，但现在都什么年代了，可以先安装默认的版本，接着再安装 `gcc-4.9`、`gcc-5` 之类的！

（注意目前 `gcc-5` 实际上是 `5.4.1`，没有 `5.1` 或 `5.2` 可供选择，已提供 `gcc-6` 版本，目前版本是 `6.1.1`。）

```shell
$ sudo apt-get upgrade   # 这句可以不执行

$ sudo apt-get install gcc-4.8 g++-4.8
$ sudo apt-get install gcc-4.9 g++-4.9
$ sudo apt-get install gcc-5 g++-5
$ sudo apt-get install gcc-6 g++-6
```

（非必须）现在可以考虑刷新一下，否则 `locate` 等命令，是找不到新版本文件所在目录的：

```shell
$ sudo updatedb && sudo ldconfig

$ locate gcc
```

你会发现 `gcc -v` 显示出来的版本还是 `gcc-4.8` 的，因此需要更新一下链接。

## 3. 设置 gcc 的链接，便于切换 ##

(保留原来的 `4.8.x` 版本，便于快速切换)

多行命令版本：

```shell
$ sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.8 48 \
--slave /usr/bin/gcc-ar gcc-ar /usr/bin/gcc-ar-4.8 \
--slave /usr/bin/gcc-nm gcc-nm /usr/bin/gcc-nm-4.8 \
--slave /usr/bin/gcc-ranlib gcc-ranlib /usr/bin/gcc-ranlib-4.8

$ sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-4.8 48

$ sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.9 49 \
--slave /usr/bin/gcc-ar gcc-ar /usr/bin/gcc-ar-4.9 \
--slave /usr/bin/gcc-nm gcc-nm /usr/bin/gcc-nm-4.9 \
--slave /usr/bin/gcc-ranlib gcc-ranlib /usr/bin/gcc-ranlib-4.9

$ sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-4.9 49

$ sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-5 54 \
--slave /usr/bin/gcc-ar gcc-ar /usr/bin/gcc-ar-5 \
--slave /usr/bin/gcc-nm gcc-nm /usr/bin/gcc-nm-5 \
--slave /usr/bin/gcc-ranlib gcc-ranlib /usr/bin/gcc-ranlib-5

$ sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-5 54

$ sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-6 60 \
--slave /usr/bin/gcc-ar gcc-ar /usr/bin/gcc-ar-6 \
--slave /usr/bin/gcc-nm gcc-nm /usr/bin/gcc-nm-6 \
--slave /usr/bin/gcc-ranlib gcc-ranlib /usr/bin/gcc-ranlib-6

$ sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-6 60
```

下面是单行命令的版本：

gcc 4.8.x：

```shell
$ sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.8 48 --slave /usr/bin/gcc-ar gcc-ar /usr/bin/gcc-ar-4.8 --slave /usr/bin/gcc-nm gcc-nm /usr/bin/gcc-nm-4.8 --slave /usr/bin/gcc-ranlib gcc-ranlib /usr/bin/gcc-ranlib-4.8

$ sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-4.8 48

```

gcc 4.9.x：

```shell
$ sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.9 49 --slave /usr/bin/gcc-ar gcc-ar /usr/bin/gcc-ar-4.9 --slave /usr/bin/gcc-nm gcc-nm /usr/bin/gcc-nm-4.9 --slave /usr/bin/gcc-ranlib gcc-ranlib /usr/bin/gcc-ranlib-4.9

$ sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-4.9 49

```

gcc 5.4：

```shell
$ sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-5 54 --slave /usr/bin/gcc-ar gcc-ar /usr/bin/gcc-ar-5 --slave /usr/bin/gcc-nm gcc-nm /usr/bin/gcc-nm-5 --slave /usr/bin/gcc-ranlib gcc-ranlib /usr/bin/gcc-ranlib-5

$ sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-5 54

```

gcc 6.1.1：

```shell
$ sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-6 61 --slave /usr/bin/gcc-ar gcc-ar /usr/bin/gcc-ar-6 --slave /usr/bin/gcc-nm gcc-nm /usr/bin/gcc-nm-6 --slave /usr/bin/gcc-ranlib gcc-ranlib /usr/bin/gcc-ranlib-6

$ sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-6 61

```

## 4. 切换 gcc 版本 ##

切换 `gcc` 的版本：

```shell
$ sudo update-alternatives --config gcc
```

有 4 个候选项可用于替换 `gcc` (提供 `/usr/bin/gcc`)。

```shell
  选择       路径            优先级  状态
------------------------------------------------------------
* 0            /usr/bin/gcc-4.9   49        自动模式
  1            /usr/bin/gcc-4.8   48        手动模式
  2            /usr/bin/gcc-4.9   49        手动模式
  3            /usr/bin/gcc-5     54        手动模式
  4            /usr/bin/gcc-6     61        手动模式
```

切换 `g++` 的版本：

```shell
$ sudo update-alternatives --config g++
```

有 4 个候选项可用于替换 `g++` (提供 `/usr/bin/g++`)。

```shell
  选择       路径            优先级  状态
------------------------------------------------------------
* 0            /usr/bin/g++-4.9   49        自动模式
  1            /usr/bin/g++-4.8   48        手动模式
  2            /usr/bin/g++-4.9   49        手动模式
  3            /usr/bin/g++-5     54        手动模式
  4            /usr/bin/g++-6     61        手动模式
```

## 5. 参考文章 ##

----------------------------------------------------------------

[http://www.cnblogs.com/BlackStorm/p/5183490.html](http://www.cnblogs.com/BlackStorm/p/5183490.html)

[http://blog.sina.com.cn/s/blog_54dd80920102vvt6.html](http://www.cnblogs.com/BlackStorm/p/5183490.html)

----------------------------------------------------------------
<.end.>
