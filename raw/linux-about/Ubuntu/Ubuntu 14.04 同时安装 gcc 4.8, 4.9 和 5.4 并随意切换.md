
# Ubuntu 14.04 同时安装 gcc 4.8, 4.9 和 5.4 并随意切换

参考自下列文章：

http://www.cnblogs.com/BlackStorm/p/5183490.html<br/>
http://blog.sina.com.cn/s/blog_54dd80920102vvt6.html

关于 `ppa:ubuntu-toolchain-r/test` 的内容介绍请查阅：

https://launchpad.net/~ubuntu-toolchain-r/+archive/ubuntu/test

在 `toolchain/test` 下已经有打包好的 `gcc`，版本有 4.x、5.0、6.0、7.0、8.0 等，用这个 `PPA` 升级 `gcc` 就可以啦！

首先添加 `ppa` 库到 Ubuntu 的 `apt-get`：

	$ sudo add-apt-repository ppa:ubuntu-toolchain-r/test
	$ sudo apt-get update

如果提示 `ppa` 没有安装, 则使用下列命令安装:

	$ sudo apt-get install software-properties-common

默认在系统中安装的是 `gcc-4.8`，但现在都什么年代了万一有奇怪的更新呢，可以先升级一下，接着就可以选择安装 `gcc-4.9`、`gcc-5` 之类的啦！（注意目前 `gcc-5` 实际上是 `5.4.0`，没有 5.2 或 5.3 可供选择）

	$ sudo apt-get upgrade

	$ sudo apt-get install gcc-4.8 g++-4.8
	$ sudo apt-get install gcc-4.9 g++-4.9
	$ sudo apt-get install gcc-5 g++-5

（非必须）现在可以考虑刷新一下，否则比如 `locate` 等命令，是找不到新版本文件所在目录的：

	$ sudo updatedb && sudo ldconfig

	$ locate gcc

你会发现 `gcc -v` 显示出来的版本还是 `gcc-4.8` 的，因此需要更新一下链接:
(保留原来的 `4.8.2` 版本，便于快速切换)

	sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.8 48 \
	--slave /usr/bin/gcc-ar gcc-ar /usr/bin/gcc-ar-4.8 \
	--slave /usr/bin/gcc-nm gcc-nm /usr/bin/gcc-nm-4.8 \
	--slave /usr/bin/gcc-ranlib gcc-ranlib /usr/bin/gcc-ranlib-4.8

	sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-4.8 48 \
	--slave /usr/bin/g++-ar g++-ar /usr/bin/g++-ar-4.8 \
	--slave /usr/bin/g++-nm g++-nm /usr/bin/g++-nm-4.8 \
	--slave /usr/bin/g++-ranlib g++-ranlib /usr/bin/g++-ranlib-4.8

	sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.9 49 \
	--slave /usr/bin/gcc-ar gcc-ar /usr/bin/gcc-ar-4.9 \
	--slave /usr/bin/gcc-nm gcc-nm /usr/bin/gcc-nm-4.9 \
	--slave /usr/bin/gcc-ranlib gcc-ranlib /usr/bin/gcc-ranlib-4.9

	sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-4.9 49 \
	--slave /usr/bin/g++-ar g++-ar /usr/bin/g++-ar-4.9 \
	--slave /usr/bin/g++-nm g++-nm /usr/bin/g++-nm-4.9 \
	--slave /usr/bin/g++-ranlib g++-ranlib /usr/bin/g++-ranlib-4.9

	sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-5 54 \
	--slave /usr/bin/gcc-ar gcc-ar /usr/bin/gcc-ar-5 \
	--slave /usr/bin/gcc-nm gcc-nm /usr/bin/gcc-nm-5 \
	--slave /usr/bin/gcc-ranlib gcc-ranlib /usr/bin/gcc-ranlib-5

	sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-5 54 \
	--slave /usr/bin/g++-ar g++-ar /usr/bin/g++-ar-5 \
	--slave /usr/bin/g++-nm g++-nm /usr/bin/g++-nm-5 \
	--slave /usr/bin/g++-ranlib g++-ranlib /usr/bin/g++-ranlib-5

4.8.5:

	sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.8 48 --slave /usr/bin/gcc-ar gcc-ar /usr/bin/gcc-ar-4.8 --slave /usr/bin/gcc-nm gcc-nm /usr/bin/gcc-nm-4.8 --slave /usr/bin/gcc-ranlib gcc-ranlib /usr/bin/gcc-ranlib-4.8

	sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-4.8 48 --slave /usr/bin/g++-ar g++-ar /usr/bin/g++-ar-4.8 --slave /usr/bin/g++-nm g++-nm /usr/bin/g++-nm-4.8 --slave /usr/bin/g++-ranlib g++-ranlib /usr/bin/g++-ranlib-4.8

4.9.4:

	sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.9 49 --slave /usr/bin/gcc-ar gcc-ar /usr/bin/gcc-ar-4.9 --slave /usr/bin/gcc-nm gcc-nm /usr/bin/gcc-nm-4.9 --slave /usr/bin/gcc-ranlib gcc-ranlib /usr/bin/gcc-ranlib-4.9

	sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-4.9 49 --slave /usr/bin/g++-ar g++-ar /usr/bin/g++-ar-4.9 --slave /usr/bin/g++-nm g++-nm /usr/bin/g++-nm-4.9 --slave /usr/bin/g++-ranlib g++-ranlib /usr/bin/g++-ranlib-4.9

5.5:

	sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-5 55 --slave /usr/bin/gcc-ar gcc-ar /usr/bin/gcc-ar-5 --slave /usr/bin/gcc-nm gcc-nm /usr/bin/gcc-nm-5 --slave /usr/bin/gcc-ranlib gcc-ranlib /usr/bin/gcc-ranlib-5

	sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-5 55 --slave /usr/bin/g++-ar g++-ar /usr/bin/g++-ar-5 --slave /usr/bin/g++-nm g++-nm /usr/bin/g++-nm-5 --slave /usr/bin/g++-ranlib g++-ranlib /usr/bin/g++-ranlib-5

切换 `gcc` 的版本:

	$ sudo update-alternatives --config gcc

有 2 个候选项可用于替换 `gcc` (路径 `/usr/bin/gcc`)。

	  选择       路径            优先级  状态
	------------------------------------------------------------
	* 0            /usr/bin/gcc-4.9   49        自动模式
	  1            /usr/bin/gcc-4.8   48        手动模式
	  2            /usr/bin/gcc-4.9   49        手动模式
  	  3            /usr/bin/gcc-5     55        手动模式

	$ sudo update-alternatives --config g++

有 2 个候选项可用于替换 `g++` (路径 `/usr/bin/g++`)。

	  选择       路径            优先级  状态
	------------------------------------------------------------
	* 0            /usr/bin/g++-4.9   49        自动模式
	  1            /usr/bin/g++-4.8   48        手动模式
	  2            /usr/bin/g++-4.9   49        手动模式
  	  3            /usr/bin/g++-5     55        手动模式

==============================================================

