
# 修改 #

## 1. 修改 sources.list ##
先备份原来的 `/etc/apt/sources.list`：

```shell
cd /etc/apt/
cp sources.list sources.list.old
```

编辑 `/etc/apt/sources.list` 文件：

```shell
vim sources.list
```

然后输入 `ggdG` 命令删除原来的所有内容，把下面的内容复制粘贴进去。

```shell
deb http://mirrors.aliyun.com/ubuntu/ xenial main
deb-src http://mirrors.aliyun.com/ubuntu/ xenial main

deb http://mirrors.aliyun.com/ubuntu/ xenial-updates main
deb-src http://mirrors.aliyun.com/ubuntu/ xenial-updates main

deb http://mirrors.aliyun.com/ubuntu/ xenial universe
deb-src http://mirrors.aliyun.com/ubuntu/ xenial universe
deb http://mirrors.aliyun.com/ubuntu/ xenial-updates universe
deb-src http://mirrors.aliyun.com/ubuntu/ xenial-updates universe

deb http://mirrors.aliyun.com/ubuntu/ xenial-security main
deb-src http://mirrors.aliyun.com/ubuntu/ xenial-security main
deb http://mirrors.aliyun.com/ubuntu/ xenial-security universe
deb-src http://mirrors.aliyun.com/ubuntu/ xenial-security universe
```

4. 终端输入 `sudo apt-get update` 命令更新镜像源。


useradd -u 544 -d /usr/testuser1  -g users -m  testuser1

useradd -m -d /home/mangos -c "MaNGOS" -U mangos
useradd -m -d /home/shines77 -c "Operations staff" -g users shines77

passwd mangos
passwd shines77

## 2. sources.list 镜像 ##

打开阿里云镜像站 [https://developer.aliyun.com/mirror/]，找到 `Ubuntu`，点击右边的“帮助”，复制如上所列的内容。(`Ubuntu 18.04` 的版本请自行复制对应版本)

# 参考文章 #

[https://blog.csdn.net/mbytes/java/article/details/86737746](https://blog.csdn.net/mbytes/java/article/details/86737746)

升级 `git` 版本:

```shell
apt-get install python-software-properties
apt-get install software-properties-common
add-apt-repository ppa:git-core/ppa
apt-get update
apt-get upgrade
apt-get install git
```

检测新的版本:

git --version
