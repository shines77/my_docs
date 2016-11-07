
Ubuntu 14.04 如果使用PPTP VPN连接
=====================================

本文介绍的是 `pptpd vpn server` 的连接方法，只介绍客户端如何连接 `VPN Server`，关于怎么搭建 `VPN Server` 请参阅别的文章。

`VPN` 连接分为 `IPSec`、`PPTP`、`L2F`、`L2TP`、`GRE` 等几种方式，这里介绍的是 `PPTP` 的连接方式。

## 1. 安装 pptp client ##

安装 `pptp-linux` 组件：

```shell
$ sudo apt-get install pptp-linux
```

`pptp-linux` 包括了 `ppp`、`pppd`、`pptpsetup` 几个组件。

## 2. 安装 pptp client ##

```shell
$ sudo pptpsetup --create pptpd --server x.x.x.x --username xxxxxxxx --password xxxxxxxx --encrypt --start
```

* `-create`：是创建的 `vpn` 连接的 `tunnel` 名称，一般默认为 `pptpd`；
* `-server`：是 `vpn server` 的域名或 `ip` 地址；
* `–username`：是 `vpn` 的账号，即用户名；
* `–password`：是 `vpn` 的密码，也可以没这个参数，命令稍后会自动询问，这样可以保证账号安全；
* `–encrypt`：表示需要加密，不必指定加密方式，命令会读取配置文件中的加密方式；
* `–start`：表示连接创建完成后立即启动连接;

（如果没写 `–start` 这个参数，而又想启动刚才定义（创建）好的 `vpn` 连接，可以使用 `sudo pon pptpd` 命令，想要断开连接的话请使用命令 `sudo poff` 。）

范例：

```shell
$ sudo pptpsetup --create pptpd --server jp.ioss.pw --username freevpnss --password 41570461 --encrypt --start
$ sudo pptpsetup --create pptpd --server 106.185.42.91 --username freevpnss --password 41570461 --encrypt --start

$ sudo pptpsetup --create free_us1 --server 69.60.121.29 --username free --password 1786 --encrypt --start
$ sudo pptpsetup --create free_us2 --server 216.104.36.238 --username free --password 2867 --encrypt --start
$ sudo pptpsetup --create free_uk --server 77.92.68.65 --username free --password 1108 --encrypt --start
```

连接信息，`VPN` 服务器域名：`jp.ioss.pw`，账号：`freevpnss`，密码：`41570461`，连接方式：`PPTP IPsec(IKEv1)`，IPSec密钥：`freevpnss` 。

看到的错误信息是：

```shell
Using interface ppp0
Connect: ppp0 <--> /dev/pts/10
anon fatal[open_callmgr:pptp.c:495]: Could not launch call manager after 3 tries.
Modem hangup
Connection terminated.

```

## 3. 其他设置 ##

### 3.1 加载 ppp_mppe 模块 ###

命令：

```shell
$ sudo modprobe ppp_mppe
```

### 3.2 启动连接 ###

命令：

```shell
$ sudo pon pptpd
或者
$ pppd call pptpd
```

其中 `pptpd` 是前面用 `pptpsetup` 创建的 `tunnel` 名称。

### 3.3 连接配置信息 ###

```shell
$ sudo vim /etc/ppp/peers/pptpd

# written by pptpsetup
pty "pptp jp.ioss.pw --nolaunchpppd"
lock
noauth
nobsdcomp
nodeflate
name freevpnss
remotename pptpd
ipparam pptpd
require-mppe-128
```

### 3.4 其他配置 ###

pap 密码：

```shell
$ sudo vim /etc/ppp/pap-secrets

*       pptpd           ""              *
```

options 选项：

```shell
$ sudo vim /etc/ppp/options.pptpd

name pptpd

# Lock the port
lock

# Authentication
# We don't need the tunnel server to authenticate itself
noauth

# We won't do PAP, EAP, CHAP, or MSCHAP, but we will accept MSCHAP-V2
# (you may need to remove these refusals if the server is not using MPPE)
refuse-pap
refuse-eap
refuse-chap
refuse-mschap

# Compression
# Turn off compression protocols we know won't be used
nobsdcomp
nodeflate

# Encryption
# (There have been multiple versions of PPP with encryption support,
# choose with of the following sections you will use.  Note that MPPE
# requires the use of MSCHAP-V2 during authentication)

# http://ppp.samba.org/ the PPP project version of PPP by Paul Mackarras
# ppp-2.4.2 or later with MPPE only, kernel module ppp_mppe.o
{{{
Require MPPE 128-bit encryption
equire-mppe-128
}}}

# http://polbox.com/h/hs001/ fork from PPP project by Jan Dubiec
# ppp-2.4.2 or later with MPPE and MPPC, kernel module ppp_mppe_mppc.o
# {{{
# Require MPPE 128-bit encryption
#mppe required,stateless
# }}}
```

密码设置：

```shell
$ sudo vim /etc/ppp/chap-secrets

# Secrets for authentication using CHAP
# client        server  secret                  IP addresses

# added by pptpsetup for pptpd
freevpnss pptpd "41570461" *
```

## 4. 关于免费的 PPTP VPN ##

请访问：[http://freevpnss.cc/](http://freevpnss.cc/) （需要先翻墙才能打开）

这个网站的 `VPN` 使用的是 `PPTP IPsec(IKEv1)` 连接方式，还算稳定，日本的节点大多数时候速度都挺快，第一个节点（美国）的速度也还可以。

也许还有许多免费的 `PPTP VPN`，请自行百度或 `Google` 。

## 5. 参考文章 ##

linux 下如何 vpn 连接呢？<br/>
[https://zhidao.baidu.com/question/375764937.html](https://zhidao.baidu.com/question/375764937.html)

ubuntu 连接 vpn 服务器(pptpd)<br/>
[http://blog.sina.com.cn/s/blog_630d9b440100fjpu.html](http://blog.sina.com.cn/s/blog_630d9b440100fjpu.html)

IPSEC L2TP VPN on Ubuntu 14.04 with OpenSwan, xl2tpd and ppp<br/>
[https://raymii.org/s/tutorials/IPSEC_L2TP_vpn_with_Ubuntu_14.04.html](https://raymii.org/s/tutorials/IPSEC_L2TP_vpn_with_Ubuntu_14.04.html)

.
