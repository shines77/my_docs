
#  Ubuntu 14.04 上如何安装 JDK 8

## 1. 简介
`Ubuntu 14.04` 安装 `JDK` 有两种方式：

1. 通过 `ppa` (源) 方式安装；
2. 通过官网下载安装包解压安装。

推荐使用第一种方式，因为可以通过 `apt-get upgrade` 方式方便的获得 `JDK` 的升级。

## 2.1 `ppa` 源方式安装

### 2.1.1 添加 `ppa` 源

```bash
sudo add-apt-repository ppa:webupd8team/java

sudo apt-get update
```

第一步需要按 `[Enter]` 键 (回车键) 确认是否添加 `ppa` 源。

### 2.1.2 安装 `oracle-java-installer`

JDK 7

```
sudo apt-get install oracle-java7-installer
```

JDK 8

```
sudo apt-get install oracle-java8-installer
```

安装器会提示你同意 `oracle` 的服务条款，第一步选择 `ok`，然后第二部选择 `yes` 即可。

如果你不想自己手动点击，也可以加入下面的这条命令，默认同意条款：

JDK 7 默认选择条款

```
echo oracle-java7-installer shared/accepted-oracle-license-v1-1 select true | sudo /usr/bin/debconf-set-selections
```

JDK 8 默认选择条款

```
echo oracle-java8-installer shared/accepted-oracle-license-v1-1 select true | sudo /usr/bin/debconf-set-selections
```

注：如果你由于某些原因，`installer` 的下载速度很慢，那么可以中断操作。

然后手动下载好相应 `JDK` 的 `tar.gz` 包，放在如下路径里：

```
/var/cache/oracle-jdk7-installer             (JDK 7) 
/var/cache/oracle-jdk8-installer             (JDK 8) 
```

然后再使用上面的命令安装一次 `installer`，`installer` 则会默认使用你下载的 `tar.gz` 安装包。

### 2.1.3 设置系统默认的 `JDK` 版本

如果你既安装了 `JDK 7`，也安装了 `JDK 8`，要实现两者的切换，可以：

切换到 `JDK 7`

```
sudo update-java-alternatives -s java-7-oracle
```

切换到 `JDK 8`

```
sudo update-java-alternatives -s java-8-oracle
```


## 2.2 下载安装包解压安装

### 2.2.1 下载 JDK 8

先去官网下载 `JDK 8`，网址如下：

```
https://www.oracle.com/technetwork/java/javase/downloads/jdk8-downloads-2133151.html
```
选择 `Java SE Development Kit 8u191` 下的 `Linux x64` 版本：

下载地址如下（由于不能直接使用 `wget` 命令直接下载，所以可以先下载到本地再上传到服务器上）：

```
https://download.oracle.com/otn-pub/java/jdk/8u191-b12/2787e4a523244c269598db4e85c51e0c/jdk-8u191-linux-x64.tar.gz
```

## 3. 错误处理

### 3.1 update 错误

如果 `sudo apt-get update` 的时候出现 `ppa` 源的更新错误。

例如，出现如下报错信息：

```
W: Failed to fetch http://ppa.launchpad.net/ubuntu-toolchain-r/test/ubuntu/dists/trusty/main/binary-i386/Packages  Hash Sum mismatch

E: Some index files failed to download. They have been ignored, or old ones used instead.
```

这是由于添加了下列 `ppa` 源造成的：

```
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
```

进入下列路径，把相应的 `ppa` 源注释掉即可：

```
cd /etc/apt/sources.list.d
vim ubuntu-toolchain-r-test-trusty.list
```

把下列 `ppa` 源的内容都注释即可，如下所示：

```
# deb http://ppa.launchpad.net/ubuntu-toolchain-r/test/ubuntu trusty main
# deb-src http://ppa.launchpad.net/ubuntu-toolchain-r/test/ubuntu trusty main
# deb-src http://ppa.launchpad.net/ubuntu-toolchain-r/test/ubuntu trusty main
```

## 4. 参考文章

1. [`Ubuntu 14.04 安装 JDK 的两种方式`](https://www.cnblogs.com/zsr0401/p/6279436.html)
   
    [https://www.cnblogs.com/zsr0401/p/6279436.html](https://www.cnblogs.com/zsr0401/p/6279436.html)

2. [`Ubuntu 14.04 安装 JDK 8`](https://blog.csdn.net/zhuxiaoping54532/article/details/70158200)

    [https://blog.csdn.net/zhuxiaoping54532/article/details/70158200](https://blog.csdn.net/zhuxiaoping54532/article/details/70158200)
