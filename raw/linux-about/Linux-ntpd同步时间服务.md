

在 `Ubuntu 14.04` 上安装和使用 `NTP` 服务
---------------------------------------------

## 1. 安装与配置 ##

先安装 `ntp`：

    # sudo apt-get install ntp

配置 `ntp`：

    # sudo vim /etc/ntp.conf

内容为：

    # /etc/ntp.conf, configuration for ntpd;
    # For more information about this file, see the man pages
    # ntp.conf(5), ntp_acc(5), ntp_auth(5), ntp_clock(5), ntp_misc(5), ntp_mon(5) for help.
    
    driftfile /var/lib/ntp/ntp.drift
    
    # Permit time synchronization with our time source, but do not
    # permit the source to query or modify the service on this system.

    # Access control configuration; see /usr/share/doc/ntp-doc/html/accopt.html for
    # details.  The web page <http://support.ntp.org/bin/view/Support/AccessRestrictions>
    # might also be helpful.
    #
    # Note that "restrict" applies to both servers and clients, so a configuration
    # that might be intended to block requests from certain clients could also end
    # up blocking replies from your own upstream servers.
    
    # By default, exchange time with everybody, but don't allow configuration.
    restrict -4 default kod nomodify notrap nopeer noquery
    restrict -6 default kod nomodify notrap nopeer noquery
    
    # Permit all access over the loopback interface.  This could
    # be tightened as well, but to do so would effect some of
    # the administrative functions.
    
    restrict 127.0.0.1
    restrict -6 ::1
    
    # Hosts on local network are less restricted.
    
    # 允许内网其他机器同步时间
    restrict 192.168.2.0 mask 255.255.255.0 nomodify notrap
    
    # Use public servers from the pool.ntp.org project.
    # Please consider joining the pool (http://www.pool.ntp.org/join.html).
    # 中国这边最活跃的时间服务器 : http://www.pool.ntp.org/zone/cn
    
    # prefer 代表优先使用的ip
    server 0.cn.pool.ntp.org prefer
    server 1.cn.pool.ntp.org prefer
    server 3.asia.pool.ntp.org prefer
    server 2.asia.pool.ntp.org prefer
    server 0.asia.pool.ntp.org prefer
    
    # Use Ubuntu's ntp server as a fallback.
    server ntp.ubuntu.com
    server 192.168.2.193

    # If you want to provide time to your local subnet, change the next line.
    # (Again, the address is an example only.)

    #broadcast 192.168.2.255 autokey            # broadcast server
    #broadcastclient                            # broadcast client
    #broadcast 224.0.1.1 autokey                # multicast server
    #multicastclient 224.0.1.1                  # multicast client
    #manycastserver 239.255.254.254             # manycast server
    #manycastclient 239.255.254.254 autokey     # manycast client
    # allow update time by the upper server
    
    # 允许上层时间服务器主动修改本机时间
    restrict 0.cn.pool.ntp.org nomodify notrap noquery
    restrict 3.asia.pool.ntp.org nomodify notrap noquery
    restrict 2.asia.pool.ntp.org nomodify notrap noquery
    restrict ntp.ubuntu.com nomodify notrap noquery
    restrict 192.168.2.193 nomodify notrap noquery
    
    # Undisciplined Local Clock. This is a fake driver intended for backup
    # and when no outside source of synchronized time is available.
    # 外部时间服务器不可用时, 以本地时间作为时间服务
    server  192.168.2.193 # local clock
    fudge   192.168.2.193 stratum 10
    
    # Enable public key cryptography.
    #crypto

    includefile /etc/ntp/crypto/pw
    
    # Key file containing the keys and key identifiers used when operating
    # with symmetric key cryptography.

    keys /etc/ntp/keys
    
    # Specify the key identifiers which are trusted.
    #trustedkey 4 8 42
    # Specify the key identifier to use with the ntpdc utility.
    #requestkey 8
    # Specify the key identifier to use with the ntpq utility.
    #controlkey 8
    # Enable writing of statistics records.
    #statistics clockstats cryptostats loopstats peerstats

配置参数和命令简单说明请参考: http://linux.vbird.org/linux_server/0440ntp.php#server_ntp.conf

配置文件修改完成，保存退出，启动服务：

    # sudo service ntp start

启动后，一般需要 `5-10` 分钟左右的时候才能与外部时间服务器开始同步时间。可以通过命令查询 `NTPD` 服务情况：

  1) 查看服务连接和监听:

    # sudo netstat -tlunp | grep ntp

    udp        0      0 192.168.1.135:123           0.0.0.0:*                               23103/ntpd
    udp        0      0 127.0.0.1:123               0.0.0.0:*                               23103/ntpd
    udp        0      0 0.0.0.0:123                 0.0.0.0:*                               23103/ntpd
    udp        0      0 fe80::6cae:8bff:fe3d:f65:123 :::*                                   23103/ntpd
    udp        0      0 fe80::6eae:8bff:fe3d:f65:123 :::*                                   23103/ntpd
    udp        0      0 ::1:123                     :::*                                    23103/ntpd
    udp        0      0 :::123                      :::*                                    23103/ntpd

  看红色加粗的地方，表示连接和监听已正确，采用 `UDP` 方式。

  2) `ntpq -p` 查看网络中的 `NTP` 服务器，同时显示客户端和每个服务器的关系：

    # sudo ntpq -p

	     remote           refid      st t when poll reach   delay   offset  jitter
	================================================================================
	 cn.pool.ntp.org 10.137.38.86     2 u   17   64    3   38.949  168.994 161.215
	 194.225.50.25   .STEP.          16 u    -   64    0    0.000    0.000   0.000
	 jp.pool.ntp.org 249.224.99.213   2 u   33   64    3   69.075  150.490 155.234
	 x.ns.gin.ntt.ne .STEP.          16 u  743   64    0    0.000    0.000   0.000
	 golem.canonical 140.203.204.77   2 u   34   64    3  241.382  173.716 150.142
	 192.168.2.193   .STEP.          16 u  356   64    0    0.000    0.000   0.000

  3) `ntpstat` 命令查看时间同步状态, 一般需要5-10分钟(Ubuntu 14.04里好像没有这个命令)：

    # sudo ntpstat

使用 `ntpdate` 跟指定的 `ip` 同步时间, 如果提示 `ntp` 已启动并占用端口, 需先停止 `ntp` 服务：

    # sudo service ntp stop

然后执行：

    # sudo ntpdate -u 0.cn.pool.ntp.org

检查 `ntpd` 服务是否启动成功了? ：

    # sudo service --status-all

如果看到 `ntp` 并且它所在的行显示 `"[+]"`, 即代表已经启动。


在 `CentOS 6.x` 上
------------------

先检查是否安装了 `ntp`：

    # sudo rpm -q ntp

    ntp-4.2.4p8-2.el6.x86_64
    
这表示已安装了, 如果没有安装, 这是空白.

如果没有安装, 则执行：

    # sudo yum install ntp

完成后，都需要配置 `NTP` 服务为自启动：

    # sudo chkconfig ntpd on

查看是否启动成功了:

    # sudo chkconfig --list ntpd

使用 `ntpdate` 跟指定的 `ip` 同步时间, 如果提示 `ntp` 已启动并占用端口, 需先停止 `ntp` 服务:

    # sudo ntpdate -u 0.cn.pool.ntp.org

其它类似于 `Ubuntu 14.04`, 请参考前面的内容。

-----------------------------------------

`ntpd` 既是服务器端，也是客户端。

作为服务器运行时：

1) 修改 `/etc/ntp.conf` 并添加：

    ```
	restrict 127.0.0.1
	restrict 192.168.1.0 mask 255.255.255.0 nomodify notrap
    ```

2) 启动 `ntpd` 服务：

    ```
    # sudo service ntpd start
    ```

作为客户端运行时：

1) 修改 `/etc/ntp.conf` 并添加：

    ```
	server 192.168.1.1
    ```

2) 启动 `ntpd` 服务：

    ```
	# sudo service ntpd start
    ```

-----------------------------------------

## 2. 参考文章:  ##

`NTP 服务及时间同步 (CentOS6.x 和 Ubuntu 14.04)`

[http://acooly.iteye.com/blog/1993484](http://acooly.iteye.com/blog/1993484)

<.end.>