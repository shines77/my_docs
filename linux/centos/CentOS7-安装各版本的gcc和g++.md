
# CentOS 7.x 里安装各个版本的 gcc

## gcc 4.8.5

```shell
sudo yum install centos-release-scl
sudo yum install devtoolset-4-binutils
scl enable devtoolset-4 bash
source scl_source enable devtoolset-4
which gcc
gcc --version
```

或者安装默认的 `gcc 4.8.5` 版本：

```shell
sudo yum install gcc gcc-c++ -y
```

## gcc 6.3.1

First you need to enable the `Software Collections`, then it's available in `devtoolset-6`:

```shell
sudo yum install centos-release-scl
sudo yum install devtoolset-6-gcc*
scl enable devtoolset-6 bash
which gcc
gcc --version
```

## gcc 7.x

First you need to enable the `Software Collections`, then it's available in `devtoolset-7`:

```shell
sudo yum install centos-release-scl
sudo yum install devtoolset-7-gcc*
scl enable devtoolset-7 bash
which gcc
gcc --version
```

更好的方法：

The best approach to use yum and update your devtoolset is to utilize the `CentOS` SCLo RH Testing repository:

```shell
yum install centos-release-scl-rh
yum --enablerepo=centos-sclo-rh-testing install devtoolset-7-gcc devtoolset-7-gcc-c++
```

Many additional packages are also available, to see them all:

```shell
yum --enablerepo=centos-sclo-rh-testing list devtoolset-7*
```

## Reference

* [`How to Install gcc 5.3 with yum on CentOS 7.2?`](https://stackoverflow.com/questions/36327805/how-to-install-gcc-5-3-with-yum-on-centos-7-2)

<.end.>