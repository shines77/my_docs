
为 Linux 的网卡 (NIC) 配置 IP 别名
=====================================

`Linux` 的 `IP Aliasing` 技术可以让一块网卡绑定多个 `IP` 地址，下面详细介绍在各个系统下的 `IP 别名` 的设置方法。

# 1. Debian / Ubuntu #

## 1.1. 临时性用法 ##

假如你的网卡 `eth0` 的 `IP` 设置是：192.168.3.100，配置文件如下：

```shell
$ sudo vim /etc/network/interfaces

auto eth0
iface eth0 inet static
address 192.168.3.100
netmask 255.255.255.0
broadcast 192.168.3.255
network 192.168.3.0
```

注：其中的 `broadcast` 和 `network` 设置不是必需的。

想要添加 `IP 别名`，很简单，执行下面命令即可：

```shell
$ sudo ifconfig eth0:0 192.168.3.101 up
```

然后使用 “`ifconfig -a | less`” 命令查看。

以此类推，你还可以继续添加别的 `IP` 地址，例如：

```shell
$ sudo ifconfig eth0:1 192.168.3.102 up
$ sudo ifconfig eth0:2 192.168.3.103 up
......
```

但是，这样做当系统重启以后，所有 `IP Alias` 配置都会失效，所以我们要使用永久有效的方法。

## 1.2. 配置 IP 别名 ##

有了前面的经验，配置 `IP 别名` 也是相当简单，即使用 `eth0:N` 的格式，`N` 为数字，可以取 0 ~ 254 的任意值，最好按顺序排列，也就是说一块网卡最多只支持 255 个 `IP 别名` 地址。

还有一点需要注意的是，`IP 别名` 只能选择 `静态IP`，是不支持动态 `IP` 设置的。而且，一般也不需要在 `IP 别名` （`eth0:0`、`eth0:1` 等）地址设置里配置网关，但 `eth0` 上可以。

示例如下：

```shell
$ sudo vim /etc/network/interfaces

# (跟 eth0 无关的配置已省略)

auto eth0
iface eth0 inet static
address 192.168.3.100
netmask 255.255.255.0
broadcast 192.168.3.255
network 192.168.3.0

auto eth0:0
iface eth0:0 inet static
address 192.168.3.101
netmask 255.255.255.0
broadcast 192.168.3.255
network 192.168.3.0

auto eth0:1
iface eth0:2 inet static
address 192.168.3.102
netmask 255.255.255.0
broadcast 192.168.3.255
network 192.168.3.0

auto eth0:2
iface eth0:2 inet static
address 192.168.3.103
netmask 255.255.255.0
broadcast 192.168.3.255
network 192.168.3.0

# 以此类推 ......
```

注：同样，其中的 `broadcast` 和 `network` 设置不是必需的。

保存文件后，重启系统让配置生效。如果不想重启系统，可以尝试下面的命令：

```shell
$ sudo /etc/init.d/networking restart
$ sudo ifup eth0:1
$ sudo ifup eth0:2
$ sudo ifup eth0:3
$ sudo ifup eth0:4

...... (以此类推，此处省略。)
```

# 2. RedHat / RHEL / CentOS / Fedora #

## 2.1 常规方法 ##

拷贝 `eth0 ` 的网络配置：

```shell
$ sudo cp /etc/sysconfig/network-scripts/ifcfg-eth0 /etc/sysconfig/network-scripts/ifcfg-eth0:0
```

编辑配置文件 `ifcfg-eth0:0` ：

```shell
$ sudo vim /etc/sysconfig/network-scripts/ifcfg-eth0:0
```

把 “`DEVICE=eth0`” 替换成 “`DEVICE=eth0:0`”，“`NAME=eth0`” 替换成 “`NAME=eth0:0`”，以及修改 `IPADDR`、`NETMASK`、`NETMASK` 等配置，并且注意要删掉 `GATEWAY` 这一项配置，因为 `IP 别名` 不支持网关。

例如：

```shell
DEVICE=eth0:0
NAME=eth0:0
IPADDR=192.168.3.101
NETMASK=255.255.255.0
NETWORK=192.168.3.0
ONBOOT=yes
```

同时打开 `/etc/sysconfig/network-scripts/ifcfg-eth0` 把里面的 `GATEWAY` 一行注释掉，例如：

```shell
# GATEWAY=192.168.3.254
```

打开一个新文件，`/etc/sysconfig/network`，把网关的设置写到这个文件里：

```shell
$ sudo vim /etc/sysconfig/network

GATEWAY=192.168.3.254
```

保存文件，并执行下面的命令，让配置生效：

```shell
$ sudo ifup eth0:0

或者

$ sudo service network restart
```

其他的 `eth0:1` 和 `eth0:2`，按照同样的方法配置即可，最多可以配置到 `eth0:254` 。

## 2.2 多IP地址范围 (Multiple IP address range) ##

在 `Redhat/RHEL/CentOS/Fedora` 系统上，你还可以通过 `Multiple IP address range` 的方式为 `eth0` 配置 `IP地址范围`，例如：

```shell
$ sudo vim /etc/sysconfig/network-scripts/ifcfg-eth0-range0
```

添加下面的内容，：

```shell
IPADDR_START=192.168.3.101
IPADDR_END=192.168.3.120
CLONENUM_START=0
NETMASK=255.255.255.0
```

保存文件，执行下面的命令重启网络，或者重启系统让配置生效。

```shell
$ sudo service network restart
```

# 3. 参考文章 #

Linux Creating or Adding New Network Alias To a Network Card (NIC)<br/>
[http://www.cyberciti.biz/faq/linux-creating-or-adding-new-network-alias-to-a-network-card-nic/](http://www.cyberciti.biz/faq/linux-creating-or-adding-new-network-alias-to-a-network-card-nic/)

Quick HOWTO : Ch03 : Linux Networking<br/>[http://www.linuxhomenetworking.com/wiki/index.php/Quick_HOWTO_:_Ch03_:_Linux_Networking#Creating_Interface_Aliases](http://www.linuxhomenetworking.com/wiki/index.php/Quick_HOWTO_:_Ch03_:_Linux_Networking#Creating_Interface_Aliases)

Debian: Multiple IP addresses on one Interface<br/>
[https://wiki.debian.org/NetworkConfiguration#Legacy_method](https://wiki.debian.org/NetworkConfiguration#Legacy_method)

Ubuntu 14.04 » Ubuntu 服务器指南 » 联网 <br/>
[https://help.ubuntu.com/14.04/serverguide/network-configuration.html](https://help.ubuntu.com/14.04/serverguide/network-configuration.html)

Ubuntu 16.04 » Ubuntu 服务器指南 » 联网 <br/>
[https://help.ubuntu.com/lts/serverguide/network-configuration.html](https://help.ubuntu.com/lts/serverguide/network-configuration.html)

Ubuntu 14.04 » Ubuntu 服务器指南 <br/>
[https://help.ubuntu.com/14.04/serverguide/index.html](https://help.ubuntu.com/14.04/serverguide/index.html)

.
