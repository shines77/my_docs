
# 1. 问题 #

## 1.1 libstdc 和 glibc ##

```bash
$ apt-get

apt-get: relocation error: /usr/lib/x86_64-linux-gnu/libapt-pkg.so.5.0: symbol _ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7compareERKS4_, version GLIBCXX_3.4.21 not defined in file libstdc++.so.6 with link time reference
```

解决方法:

先下载 `Ubuntu 16.04` 的 `libstdc` deb 包，再用 "`sudo dpkg -i`" 命令安装。

```bash
$ wget -c http://security.ubuntu.com/ubuntu/pool/main/g/gcc-5/libstdc++6_5.4.0-6ubuntu1~16.04.12_amd64.deb

$ sudo dpkg -i libstdc++6_5.4.0-6ubuntu1~16.04.12_amd64.deb
```

查看一下，可以看到 `libstdc++6` 已经更新为 `16.04.12` 版本：

```bash
$ dpkg -l | grep libstd

ii  libstdc++-4.8-dev:amd64                 4.8.5-4ubuntu8~14.04.2
ii  libstdc++-4.9-dev:amd64                 4.9.4-2ubuntu1~14.04.1
ii  libstdc++-5-dev:amd64                   5.5.0-12ubuntu1~14.04
ii  libstdc++-6-dev:amd64                   6.5.0-2ubuntu1~14.04.1
iU  libstdc++6:amd64                        5.4.0-6ubuntu1~16.04.12
```

## 1.2 apt-get 的问题 ##

上面的方法暂时缓解了 `apt-get` 不能使用的情况，但是很多 `Package` 依然还是 `14.04` 的，使用 “`apt install xxxxxx`” 命令时，依然会提示安装包依赖错误。

要解决这个问题，要使用 “`apt -f install`” 命令，这会移除那些包依赖错误的安装包，但同时也会带来一个新的问题，那就是 `apt` 命令也不可用了！

下面我们要解决这个问题，先下载一个 `apt` 的 deb 安装包：

```bash
cd /home/downloads

$ wget -c http://mirrors.163.com/ubuntu/pool/main/a/apt/apt_2.1.4_amd64.deb
$ sudo dpkg -i apt_2.1.4_amd64.deb

$ wget -c http://mirrors.163.com/ubuntu/pool/main/a/apt/libapt-pkg6.0_2.1.4_amd64.deb
$ sudo dpkg -i libapt-pkg6.0_2.1.4_amd64.deb

$ wget -c http://mirrors.163.com/ubuntu/pool/main/a/apt/libapt-pkg-dev_2.1.4_amd64.deb
$ sudo dpkg -i libapt-pkg-dev_2.1.4_amd64.deb

$ wget -c http://mirrors.163.com/ubuntu/pool/main/a/apt/apt-utils_2.1.4_amd64.deb
$ sudo dpkg -i apt-utils_2.1.4_amd64.deb
```

```shell
http://mirrors.163.com/ubuntu/pool/main/g/gcc-6/gcc-6_6.4.0.orig.tar.gz
```

# 2. 参考文章 #

1. [`apt-get: relocation error: /usr/lib/x86_64-linux-gnu/libapt-pkg.so.5.0: symbol _ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7compareERKS4_, version GLIBCXX_3.4.21 not defined in file libstdc++.so.6 with link time reference #4164`](https://github.com/scylladb/scylla/issues/4164)
   
   [https://github.com/scylladb/scylla/issues/4164](https://github.com/scylladb/scylla/issues/4164)

2. [`apt: relocation error: version GLIBCXX_3.4.21 not defined in file libstdc++.so.6 with link time reference`](https://askubuntu.com/questions/777803/apt-relocation-error-version-glibcxx-3-4-21-not-defined-in-file-libstdc-so-6)
   
   [https://askubuntu.com/questions/777803/apt-relocation-error-version-glibcxx-3-4-21-not-defined-in-file-libstdc-so-6](https://askubuntu.com/questions/777803/apt-relocation-error-version-glibcxx-3-4-21-not-defined-in-file-libstdc-so-6)


