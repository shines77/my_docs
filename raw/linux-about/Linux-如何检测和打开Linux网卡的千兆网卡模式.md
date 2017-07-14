
如何检测和打开Linux网卡的千兆网卡模式
------------------------------------

## 1. 首先要安装 ethtool ##

ArchLinux 下安装 ethtool:

    $ sudo pacman -S ethtool

Ubuntu 下安装 ethtool:

    $ sudo apt-get install ethtool

CentOS 下安装 ethtool:

	$ sudo yum install ethtool

## 2. 查询你的主网卡硬件信息 ##

例如, 你的主网卡是 eth0, 可以用下面的命令查询网卡的硬件信息:

    $ sudo ethtool eth0

    Settings for eth0:
        Supported ports: [ TP MII ]
        Supported link modes:   10baseT/Half 10baseT/Full
                                100baseT/Half 100baseT/Full
                                1000baseT/Half 1000baseT/Full
        Supported pause frame use: No
        Supports auto-negotiation: Yes
        Advertised link modes:  10baseT/Half 10baseT/Full
                                100baseT/Half 100baseT/Full
        Advertised pause frame use: Symmetric Receive-only
        Advertised auto-negotiation: Yes
        Link partner advertised link modes:  10baseT/Half 10baseT/Full
                                             100baseT/Half 100baseT/Full
        Link partner advertised pause frame use: No
        Link partner advertised auto-negotiation: Yes
        Speed: 100Mb/s
        Duplex: Full
        Port: MII
        PHYAD: 0
        Transceiver: internal
        Auto-negotiation: on
        Supports Wake-on: pumbg
        Wake-on: g
        Current message level: 0x00000033 (51)
                       drv probe ifdown ifup
        Link detected: yes

从上可以看到 eth0 目前是 100Mb/s 的:

	Speed: 100Mb/s,

其中 Advertised link modes 里显示不支持 1000Mb/s, 但 Supported link modes 里可以看到是支持 1000Mb/s 的, 要如何设置成千兆带宽呢?

## 3. 设置网卡的速率 ##

Linux/Unix 命令之 Ethtool (设置千兆网卡速度及模式)

描述：

    Ethtool 是用于查询及设置网卡参数的命令。

概要：

    ethtool ethX       // 查询 ethX 网口基本设置
    ethtool –h         // 显示 ethtool 的命令帮助(help)
    ethtool –i ethX    // 查询 ethX 网口的相关信息
    ethtool –d ethX    // 查询 ethX 网口注册性信息
    ethtool –r ethX    // 重置 ethX 网口到自适应模式
    ethtool –S ethX    // 查询 ethX 网口收发包统计

    ethtool –s ethX [speed 10|100|1000]\        // 设置网口速率10M/100M/1000M
    [duplex half|full]\                         // 设置网口半/全双工
    [autoneg on|off]\                           // 设置网口是否自协商
    [port tp|aui|bnc|mii]\                      // 设置网口类型
    [phyad N]\
    [xcvr internal|exteral]\
    [wol p|u|m|b|a|g|s|d...]\
    [sopass xx:yy:zz:aa:bb:cc]\
    [msglvl N]

设置成 1000Mb/s 速率:

    $ sudo ethtool -s eth0 speed 1000

如果这条命令不OK的话, 则要使用自协商检测的方式, 如下:

    $ sudo ethtool -s eth0 speed 1000 duplex full autoneg on

设置正确以后的 sudo ethtool eth0 信息如下:

	$ sudo ethtool eth0

	Settings for eth0:
		Supported ports: [ TP MII ]
		Supported link modes:   10baseT/Half 10baseT/Full
								100baseT/Half 100baseT/Full
								1000baseT/Half 1000baseT/Full
		Supported pause frame use: No
		Supports auto-negotiation: Yes
		Advertised link modes:  1000baseT/Full
		Advertised pause frame use: Symmetric Receive-only
		Advertised auto-negotiation: Yes
		Link partner advertised link modes:  10baseT/Half 10baseT/Full
											 100baseT/Half 100baseT/Full
											 1000baseT/Full
		Link partner advertised pause frame use: No
		Link partner advertised auto-negotiation: Yes
		Speed: 1000Mb/s
		Duplex: Full
		Port: MII
		PHYAD: 0
		Transceiver: internal
		Auto-negotiation: on
		Supports Wake-on: pumbg
		Wake-on: g
		Current message level: 0x00000033 (51)
					   drv probe ifdown ifup
		Link detected: yes

(注: 每一块网卡显示的信息和 Port 的类型都是不一样的, 所以可能会有些出入.)

可以看到, Speed: 1000Mb/s, 大功告成! 设置成功以后, 你的网卡会被重置, SSH 会被重连!

**可能会遇到的问题**

使用 100M/1000M 自协商也可能会检测成为 100Mb/s, 看了下面这篇文章, 里面说, 设置失败以后, 你可以尝试重启一下机器, 然后就可以检测成为 1000Mb/s 了:

[http://whxhz.iteye.com/blog/1342383](http://whxhz.iteye.com/blog/1342383)

## 4. 怎样使ethtool设置永久保存在网络设备中？##

**a. 解决方法一**

ethtool 设置可通过 /etc/sysconfig/network-scripts/ifcfg-ethX 文件保存, 从而在设备下次启动时激活选项。

例如：

    $ sudo ethtool -s eth0 speed 1000 duplex full autoneg on

此指令将 eth0 设备设置为全双工自适应，速度为1000Mbs。若要 eth0 启动时设置这些参数, 修改文件 /etc/sysconfig/network-scripts/ifcfg-eth0 ，添加如下一行:

    ETHTOOL_OPTS="speed 1000 duplex full autoneg on"

    (以上方法在 ArchLinux 上不适用, Ubuntu上未测试)

**b. 解决方法二**

将 ethtool 设置写入 /etc/rc.d/rc.local 之中。
(注: 这个方法在 ArchLinux 上一样行不通, 可以试试写入 /etc/profile 文件中.)

    $ sudo vi /etc/rc.d/rc.local

加入下面三行:

	if [ $(id -u) = "0" ]; then
			ethtool -s eth0 speed 1000 duplex full autoneg on
	fi

## 5. 参考文章 ##

Linux/Unix命令之Ethtool (设置千兆网卡速度及模式)

[http://www.cnblogs.com/gergro/archive/2008/09/17/1292730.html](http://www.cnblogs.com/gergro/archive/2008/09/17/1292730.html)

LINUX查看网卡带宽（100M还是1000M）的命令

[http://www.51testing.com/html/90/360490-849108.html](http://www.51testing.com/html/90/360490-849108.html)
