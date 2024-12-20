
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
sudo apt-get install gcc-snapshot

sudo apt-get update
```

其中第二句执行的时候，如果你已经安装过 `add-apt-repository` 了的话，按回车确认，继续即可。

如果出现下面的错误信息或其他错误提示，则说明 `add-apt-repository` 没有安装或无法正常工作：

```shell
sudo: add-apt-repository: command not found
```

注意：在添加了 `ppa` 源以后，一定要记得执行 `sudo apt-get update` 命令，才会让添加的 `ppa` 生效。

注：由于 Ubuntu 20.04 版本或更高的版本，添加 `ppa:ubuntu-toolchain-r/test` 会失败，故可以使用手动添加源地址的方法。

关于 `ubuntu-toolchain-r` 的信息可以来这里查阅：[Ubuntu - ToolChain](https://wiki.ubuntu.com/ToolChain) 。

找到其中 `PPA packages` 的部分，有如下的链接：

```shell
https://launchpad.net/~ubuntu-toolchain-r/+archive/ubuntu/test
```

打开上面的链接，展开页面中的 `Technical details about this PPA`，有一个 `Display sources.list entries for:` 的下拉列表，选择你当前 Ubuntu 的版本，例如是 `Focal(20.04)`，即可得到：

```shell
deb https://ppa.launchpadcontent.net/ubuntu-toolchain-r/test/ubuntu focal main
deb-src https://ppa.launchpadcontent.net/ubuntu-toolchain-r/test/ubuntu focal main
```

Signing key (签名的 key，后面可能会用到)：

```text
4096R/C8EC952E2A0E1FBDC5090F6A2C277A0A352154E5
```

由于 `https://ppa.launchpadcontent.net` 这个源，国内的服务器访问不了，所以我们更换为国内能访问的镜像源，例如：

```shell
deb https://launchpad.proxy.ustclug.org/ubuntu-toolchain-r/test/ubuntu focal main
deb-src https://launchpad.proxy.ustclug.org/ubuntu-toolchain-r/test/ubuntu focal main
```

其他的镜像源还有：

```bash
# 中科院镜像源
https://launchpad.proxy.ustclug.org

# 清华大学镜像源
https://mirrors.tuna.tsinghua.edu.cn

# 阿里云镜像源 (阿里云好像没有 gcc 的 ppa)
https://mirrors.aliyun.com
或者
http://mirrors.cloud.aliyuncs.com
```

把 `https://ppa.launchpadcontent.net` 更换为上面的某一个镜像源即可。

用 `vim` 新建一个源 list 文件，例如：

```
sudo vim /etc/apt/sources.list.d/ubuntu-toolchain-r-ubuntu-test-focal.list
```

把上面的两条源地址写到这个源 list 文件里，保存退出。

添加完源以后，使用 `apt update` 命令更新一下源，看源是否能访问。如果提示需要输入 key 之类的就说明能访问，否则更新会卡住并报错。

（注：如果更新成功，并且没有提示你输入 key，则可跳过下面这一步。）

如果能访问的话，并提示要输入 key 之类的，则重新执行添加 `ppa` 的命令：

```bash
$ sudo add-apt-repository ppa:ubuntu-toolchain-r/test

 Toolchain test builds; see https://wiki.ubuntu.com/ToolChain

 More info: https://launchpad.net/~ubuntu-toolchain-r/+archive/ubuntu/test
Press [ENTER] to continue or Ctrl-c to cancel adding it.
```

他会问你，按回车继续？或者 `Ctrl + C` 取消？按回车即可，这个时候 `ppa` 就添加成功了。

最后，执行 `apt update` 更新一下，即可安装 `gcc-11`、`gcc-12` (注：好像 gcc 12.x 安装的名字不叫这个，可以自己研究一下)、`gcc-13` 等等，目前已支持到 `gcc-15` (截止 2024年11月18日)。

建议也可以执行一下 `apt upgrade` ，因为新添加了源，系统也会更新一些东西。

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
sudo apt-get install gcc-10 g++-10
sudo apt-get install gcc-11 g++-11
......
# (截止到 2024年11月18日 已支持到 gcc-15 了)
sudo apt-get install gcc-15 g++-15
```

（注意：`gcc-4.8` 安装的版本是 `4.8.5`，比 `Ubuntu 14.04` 系统默认安装的版本 `4.8.4` 略高。 `gcc-5` 目前已经更新到了 `5.5.0`，`gcc-6` 目前则已经更新到了 `6.5.0` 版本。最后验证日期：`2021` 年 `11` 月 `7` 日。）

现在要刷新一下系统数据和设置，该步骤最好做一下（推荐），否则在使用 `locate`, `which` 等命令时，是搜索不到上面更新的 `clang` 相关的文件或目录的：

```shell
sudo updatedb && sudo ldconfig

locate gcc
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

`gcc 9.4.0`：

```shell
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 94 --slave /usr/bin/gcc-ar gcc-ar /usr/bin/gcc-ar-9 --slave /usr/bin/gcc-nm gcc-nm /usr/bin/gcc-nm-9 --slave /usr/bin/gcc-ranlib gcc-ranlib /usr/bin/gcc-ranlib-9 --slave /usr/bin/g++ g++ /usr/bin/g++-9
```

`gcc 10.5.0`：(要求 `Ubuntu 18.04`)

```shell
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 105 --slave /usr/bin/gcc-ar gcc-ar /usr/bin/gcc-ar-10 --slave /usr/bin/gcc-nm gcc-nm /usr/bin/gcc-nm-10 --slave /usr/bin/gcc-ranlib gcc-ranlib /usr/bin/gcc-ranlib-10 --slave /usr/bin/g++ g++ /usr/bin/g++-10
```

`gcc 11.1.0`：(要求 `Ubuntu 18.04`)

```shell
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 111 --slave /usr/bin/gcc-ar gcc-ar /usr/bin/gcc-ar-11 --slave /usr/bin/gcc-nm gcc-nm /usr/bin/gcc-nm-11 --slave /usr/bin/gcc-ranlib gcc-ranlib /usr/bin/gcc-ranlib-11 --slave /usr/bin/g++ g++ /usr/bin/g++-11
```

`gcc 11.4.0`：(要求 `Ubuntu 20.04`)

```shell
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 114 --slave /usr/bin/gcc-ar gcc-ar /usr/bin/gcc-ar-11 --slave /usr/bin/gcc-nm gcc-nm /usr/bin/gcc-nm-11 --slave /usr/bin/gcc-ranlib gcc-ranlib /usr/bin/gcc-ranlib-11 --slave /usr/bin/g++ g++ /usr/bin/g++-11
```

`gcc 13.1.0`：(要求 `Ubuntu 20.04`)

```shell
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-13 131 --slave /usr/bin/gcc-ar gcc-ar /usr/bin/gcc-ar-13 --slave /usr/bin/gcc-nm gcc-nm /usr/bin/gcc-nm-13 --slave /usr/bin/gcc-ranlib gcc-ranlib /usr/bin/gcc-ranlib-13 --slave /usr/bin/g++ g++ /usr/bin/g++-13
```

## 4. 切换 gcc 版本

### 4.1. 切换 `gcc` 版本的命令

```shell
sudo update-alternatives --config gcc
```

例如，有 7 个候选项可用于替换 `gcc` （默认路径 `/usr/bin/gcc`），如下所示：

```shell
  选择       路径            优先级  状态
------------------------------------------------------------
* 0            /usr/bin/gcc-4.9   49        自动模式
  1            /usr/bin/gcc-4.8   48        手动模式 (manual mode)
  2            /usr/bin/gcc-4.9   49        手动模式 (manual mode)
  3            /usr/bin/gcc-5     55        手动模式 (manual mode)
  4            /usr/bin/gcc-6     65        手动模式 (manual mode)
  4            /usr/bin/gcc-7     75        手动模式 (manual mode)
  5            /usr/bin/gcc-8     84        手动模式 (manual mode)
  6            /usr/bin/gcc-9     94        手动模式 (manual mode)
```

注：为了方便，切换 `gcc` 的同时，我们也切换到相同版本的 `g++`。

如果想单独切换 `g++` 的版本，可以参考 `gcc` 的版本，修改上一小节的 `update-alternatives` 配置。

同理，`alternatives` 除了可以管理 `gcc`，`g++` 以外，任何其他系列工具软件都可以使用 `alternatives` 来管理和切换，比如：`clang`。

## 5. 参考文章

----------------------------------------------------------------

* [http://www.cnblogs.com/BlackStorm/p/5183490.html](http://www.cnblogs.com/BlackStorm/p/5183490.html)

* [http://blog.sina.com.cn/s/blog_54dd80920102vvt6.html](http://www.cnblogs.com/BlackStorm/p/5183490.html)

----------------------------------------------------------------

## 6. 更新历史

* `2024` / `11` / `18` ：更新了 `ppa:ubuntu-toolchain-r/test` 源访问不了的解决办法，支持到 `Ubuntu 20.04`。

* `2021` / `11` / `07` ：更新到 `gcc 11.1.0`，但 `Ubuntu 16.04` 最高只支持到 `gcc 9.4`。

* `2020` / `09` / `12` ：把 `gcc`, `g++` 的 `alternatives` 合并在一起了。

* `2020` / `08` / `31` ：更新到 Ubuntu 16.04，并新增 `gcc` 7.5, 8.4, 9.4 等版本。

* `2018` / `07` / `19` ：原始版本

<.end.>
