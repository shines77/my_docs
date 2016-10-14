
为 Linux 的网卡 (NIC) 配置 IP 别名
================================

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

修改好配置以后，重启系统让配置生效。

# X. 参考文章 #

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
