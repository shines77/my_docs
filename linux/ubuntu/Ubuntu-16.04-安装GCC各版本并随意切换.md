
# Ubuntu 16.04 安装 GCC 各个版本并随意切换

## 1. 添加 PPA 源

在 `toolchain/test` 下已经有打包好的 `gcc` ，版本有 `4.x`、`5.0`、`6.0` 等，用这个 `PPA` 升级 `gcc` 就可以啦！

首先添加 `ppa` 到更新源，并更新 `apt-get` ：

```shell
# 安装 add-apt-repository 组件
sudo apt-get install build-essential software-properties-common -y

# 添加 ppa 源
sudo add-apt-repository ppa:ubuntu-toolchain-r/test

sudo apt-get update

# 安装 gcc 快照
apt-get install gcc-snapshot

sudo apt-get update
```

其中第二句执行的时候，如果你已经安装过 `add-apt-repository` 了的话，按回车确认，继续即可。

如果出现下面的错误信息或其他错误提示，则说明 `add-apt-repository` 没有安装或无法正常工作：

```shell
sudo: add-apt-repository: command not found
```

注意：在添加了 `ppa` 源以后，一定要记得执行 `sudo apt-get update` 命令，才会让添加的 `ppa` 生效。

## 2. 安装 gcc

`Ubuntu 14.04` 系统默认安装的版本是 `gcc-4.8`，`Ubuntu 16.04` 系统默认安装的版本是 `gcc-5.4`，过于老旧，可以先安装默认的版本，接着再安装 `gcc-6`、`gcc-7` 等等！

如果原来已经做过 `gcc` 和 `g++` 的 `update-alternatives`，并想保留原来的设置，则该步骤可以跳过。

但如果你觉得原先的设置存在问题，可以用下面的命令清除原来的设置。

删除所有的 `/usr/bin/gcc` 和 `/usr/bin/g++` 的重定向 `link`：

```shell
sudo update-alternatives --remove-all gcc
sudo update-alternatives --remove-all g++
```

然后，分别安装各个版本的 `gcc` 和 `g++`：

```shell
sudo apt-get update

sudo apt-get install gcc-4.8 g++-4.8
sudo apt-get install gcc-4.9 g++-4.9
sudo apt-get install gcc-5 g++-5
sudo apt-get install gcc-6 g++-6
sudo apt-get install gcc-7 g++-7
sudo apt-get install gcc-8 g++-8
sudo apt-get install gcc-9 g++-9
```

（注意：`gcc-4.8` 安装的版本是 `4.8.5`，比 `Ubuntu 14.04` 系统默认安装的版本 `4.8.4` 略高。 `gcc-5` 目前已经更新到了 `5.5.0`，`gcc-6` 目前则已经更新到了 `6.5.0` 版本。最后验证日期：`2020` 年 `8` 月 `31` 日。）

现在要刷新一下系统数据和设置，该步骤最好做一下（推荐），否则在使用 `locate`, `which` 等命令时，是搜索不到上面更新的 `clang` 相关的文件或目录的：

```shell
$ sudo updatedb && sudo ldconfig

$ locate gcc
```

现在，你会发现 `gcc -v` 显示出来的版本是前面安装命令里装过的 `gcc` 的最新版本，如果我们想随意切换 `gcc` 的版本，则需要更新一下软链接，我们使用 `alternatives` 来管理 `gcc`，`g++` 的软链接，方法见下一小节。

## 3. 配置 alternatives

下面是使用 `alternatives` 配置 `gcc` 的软链接的命令，以便切换。

### 3.1. 添加 alternatives 配置

`gcc 4.8.x`：

```shell
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.8 48 --slave /usr/bin/gcc-ar gcc-ar /usr/bin/gcc-ar-4.8 --slave /usr/bin/gcc-nm gcc-nm /usr/bin/gcc-nm-4.8 --slave /usr/bin/gcc-ranlib gcc-ranlib /usr/bin/gcc-ranlib-4.8 --slave /usr/bin/g++ g++ /usr/bin/g++-4.8
```

`gcc 4.9.x`：

```shell
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.9 49 --slave /usr/bin/gcc-ar gcc-ar /usr/bin/gcc-ar-4.9 --slave /usr/bin/gcc-nm gcc-nm /usr/bin/gcc-nm-4.9 --slave /usr/bin/gcc-ranlib gcc-ranlib /usr/bin/gcc-ranlib-4.9 --slave /usr/bin/g++ g++ /usr/bin/g++-4.9
```

`gcc 5.5.0`：

```shell
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-5 55 --slave /usr/bin/gcc-ar gcc-ar /usr/bin/gcc-ar-5 --slave /usr/bin/gcc-nm gcc-nm /usr/bin/gcc-nm-5 --slave /usr/bin/gcc-ranlib gcc-ranlib /usr/bin/gcc-ranlib-5 --slave /usr/bin/g++ g++ /usr/bin/g++-5
```

`gcc 6.5.0`：

```shell
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-6 65 --slave /usr/bin/gcc-ar gcc-ar /usr/bin/gcc-ar-6 --slave /usr/bin/gcc-nm gcc-nm /usr/bin/gcc-nm-6 --slave /usr/bin/gcc-ranlib gcc-ranlib /usr/bin/gcc-ranlib-6 --slave /usr/bin/g++ g++ /usr/bin/g++-6
```

`gcc 7.5.0`：

```shell
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 75 --slave /usr/bin/gcc-ar gcc-ar /usr/bin/gcc-ar-7 --slave /usr/bin/gcc-nm gcc-nm /usr/bin/gcc-nm-7 --slave /usr/bin/gcc-ranlib gcc-ranlib /usr/bin/gcc-ranlib-7 --slave /usr/bin/g++ g++ /usr/bin/g++-7
```

`gcc 8.4.0`：

```shell
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 84 --slave /usr/bin/gcc-ar gcc-ar /usr/bin/gcc-ar-8 --slave /usr/bin/gcc-nm gcc-nm /usr/bin/gcc-nm-8 --slave /usr/bin/gcc-ranlib gcc-ranlib /usr/bin/gcc-ranlib-8 --slave /usr/bin/g++ g++ /usr/bin/g++-8
```

`gcc 9.3.0`：

```shell
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 93 --slave /usr/bin/gcc-ar gcc-ar /usr/bin/gcc-ar-9 --slave /usr/bin/gcc-nm gcc-nm /usr/bin/gcc-nm-9 --slave /usr/bin/gcc-ranlib gcc-ranlib /usr/bin/gcc-ranlib-9 --slave /usr/bin/g++ g++ /usr/bin/g++-9
```

## 4. 切换 gcc 版本

### 4.1. 切换 `gcc` 版本的命令

```shell
sudo update-alternatives --config gcc
```

例如，有 5 个候选项可用于替换 `gcc` （默认路径 `/usr/bin/gcc`），如下所示：

```shell
  选择       路径            优先级  状态
------------------------------------------------------------
* 0            /usr/bin/gcc-4.9   49        自动模式
  1            /usr/bin/gcc-4.8   48        手动模式
  2            /usr/bin/gcc-4.9   49        手动模式
  3            /usr/bin/gcc-5     55        手动模式
  4            /usr/bin/gcc-6     65        手动模式
```

注：为了方便，切换 `gcc` 的同时，我们也切换到相同版本的 `g++`。

如果想单独切换 `g++` 的版本，可以参考 `gcc` 的版本，修改上一小节的 `update-alternatives` 配置。

同理，`alternatives` 除了可以管理 `gcc`，`g++` 以外，任何其他系列工具软件都可以使用 `alternatives` 来管理和切换，比如：`clang`。

## 5. 参考文章

----------------------------------------------------------------

* [http://www.cnblogs.com/BlackStorm/p/5183490.html](http://www.cnblogs.com/BlackStorm/p/5183490.html)

* [http://blog.sina.com.cn/s/blog_54dd80920102vvt6.html](http://www.cnblogs.com/BlackStorm/p/5183490.html)

----------------------------------------------------------------

## 6. 更新历史：

* `2020` / `09` / `12` ：把 `gcc`, `g++` 的 `alternatives` 合并在一起了。

* `2020` / `08` / `31` ：更新到 Ubuntu 16.04，并新增 `gcc` 7.5, 8.4, 9.3 等版本。

* `2018` / `07` / `19` ：原始版本

<.end.>
