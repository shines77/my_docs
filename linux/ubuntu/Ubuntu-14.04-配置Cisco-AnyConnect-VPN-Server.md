
Ubuntu 14.04 配置 Cisco AnyConnect VPN Server
========================================================

# 1. 简介 #

`Cisco AnyConnect VPN` 是 `Cisco（思科公司）` 的安全远程接入解决方案，隐蔽性要更好一些，穿透性和稳定性也不错，一般不太会掉线。传统的 `VPN` (LPTP, PPTP, IPSec等方式)，以及 `OpenVPN`，受到干扰可能性比较大，比较容易被 `GFW` 探知，如果流量大的话还会被 `GFW` 直接屏蔽。

`Cisco AnyConnect VPN` 还具有下列优势：

* 待机不会自动断开；
* 能够下发路由表给客户端；
* 稳定，隐蔽性好，穿透性强；
* 轻量级，速度快；
* 客户端使用简单；

# 2. 安装 ocserv (OpenConnect Server) #

`ocserv` 是一个 `OpenConnect SSL VPN` 协议服务端，从 `0.3.0` 版后开始兼容 `Cisco AnyConnect SSL VPN` 协议的终端。

官方主页：[http://www.infradead.org/ocserv/](http://www.infradead.org/ocserv/)

（注：本文是基于 `Ubuntu 14.04 x64` 而写的）

## 2.1 下载 ocserv ##

 由于 `Ubuntu 14.04` 没有提供 `ocserv` 的源安装，所以我们要下载源码然后自己编译安装，这里使用的版本是 `ocserv 0.9.2`，最新的版本是 `0.11.8` ，但依赖的组件不一样，`0.9.2` 也挺好用，有兴趣的朋友可以自行研究一下 `0.11.8` 如何编译安装。

 从官网下载 `ocserv` 的源码包，并解压：

 ```shell
$ cd ~/
$ mkdir ocserv
$ cd ocserv
$ wget ftp://ftp.infradead.org/pub/ocserv/ocserv-0.9.2.tar.xz
$ tar -xf ocserv-0.9.2.tar.xz
$ cd ocserv-0.9.2
 ```

## 2.2 依赖组件 ##

接下来，安装 `ocserv` 的依赖组件：

```bash
$ sudo apt-get install build-essential pkg-config libgnutls28-dev libwrap0-dev libpam0g-dev libseccomp-dev libreadline-dev libnl-route-3-dev
```

## 2.3 编译安装 ##

开始配置和编译源码，然后安装 `ocserv`：

```shell
$ ./configure
$ make
$ make install
```

# 3. 配置 ocserv #

## 3.1 安装证书工具 ##

先安装一下证书生成工具（`certtool`），并创建一个目录，来存放生成的证书：

```shell
$ cd ~
$ sudo apt-get install gnutls-bin
$ mkdir certificates
$ cd certificates
```

（其中的 `gnutls-bin` 安装包中包含 `certtool` 工具。）

## 3.2 CA 证书 ##

我们创建一个 `CA` 模板文件（`ca.tmpl`），其中内容如下，你可以修改其中的 `cn` 和 `organization` 字段：

```bash
$ vim ca.tmpl

cn = "VPN CA"
organization = "Cisco Inc."
serial = 1
expiration_days = 3650
ca
signing_key
cert_signing_key
crl_signing_key
```

然后，生成一个随机的 `CA key`，并用这个 `key` 和 `ca.tmpl` 模板生成 `CA 证书`（`ca-cert.pem`），如下：

```shell
$ certtool --generate-privkey --outfile ca-key.pem
$ certtool --generate-self-signed --load-privkey ca-key.pem --template ca.tmpl --outfile ca-cert.pem
```

## 3.2 Server 证书 ##

接下来，我们创建一个 `server` 证书模板（`server.tmpl`），内容如下。请注意其中的 `cn` 字段，它必须是你的服务器的域名或者 `IP` 地址。

```bash
$ vim server.tmpl

cn = "Your server's domain name or ip address"
organization = "Your Company Name"
expiration_days = 3650
signing_key
encryption_key
tls_www_server
```

例如，修改成这样，如果你的服务器没有域名则，则 `cn` 字段必须使用服务器的 `IP` 地址：

```bash
cn = "202.103.108.20"
organization = "Microsoft Inc."
或者
cn = "www.example.com"
organization = "Cisco AnyConnect"
```

（注：这里的 `organization` 字段可以跟 `ca.tmpl` 里的 `organization` 名字不一样。）

然后，生成 `server key`，并使用这个 `server key`、`CA key`、`CA 证书` 以及 `server.tmpl` 模板生成 `server 证书`（`server-cert.pem`），如下：

```shell
$ certtool --generate-privkey --outfile server-key.pem
$ certtool --generate-certificate --load-privkey server-key.pem --load-ca-certificate ca-cert.pem --load-ca-privkey ca-key.pem --template server.tmpl --outfile server-cert.pem
```

## 3.3 拷贝证书 ##

新建一个 `/etc/ocserv` 目录，拷贝 `server 证书` 和 `server key` 到 `/etc/ocserv` 目录下：

```shell
$ sudo mkdir -p /etc/ocserv
$ sudo cp server-cert.pem server-key.pem /etc/ocserv
```

## 3.4 ocserv 配置文件 ##

`ocserv` 源码里有一个简单的配置范例文件：`/ocserv-0.9.2/doc/sample.config`，把它复制到 `/etc/ocserv/` 目录下：

```shell
$ sudo cp ~/ocserv-0.9.2/doc/sample.config /etc/ocserv/ocserv.conf
```

编辑这个配置文件 `/etc/ocserv/ocserv.conf`，找到与下文相同的配置选项，并修改成下文里展示的内容，找到那几个关于 `route` 的配置项，像下面展示的一样，把它们都注释掉。最后一项 “`cisco-client-compat = true`” 在文件比较后面的地方，原本是被注释了的，将其注释去掉。

```bash
$ cd /etc/ocserv
$ sudo vim ./ocserv.conf

# ocserv 支持多种认证方式，这是自带的密码认证，使用 ocpasswd 创建密码文件
# ocserv 还支持证书认证，可以通过 Pluggable Authentication Modules (PAM) 使用 radius 等认证方式
auth = "plain[/etc/ocserv/ocpasswd]"

# 默认是 false, 修改为 true
try-mtu-discovery = true

# 证书路径
server-cert = /etc/ocserv/server-cert.pem
server-key = /etc/ocserv/server-key.pem

# 最大用户数量
max-clients = 16

# 同一个用户最多同时登陆数
max-same-clients = 10

# tcp 和 udp 端口
tcp-port = 4433
udp-port = 4433

# 运行用户和组
run-as-user = ocserv
run-as-group = ocserv

# DNS 设置
dns = 8.8.8.8
dns = 8.8.4.4

# 注释掉 route 的字段，这样表示所有流量都通过 VPN 发送
#route = 10.10.10.0/255.255.255.0
#route = 192.168.0.0/255.255.0.0
#route = fef4:db8:1000:1001::/64
#no-route = 192.168.5.0/255.255.255.0

cisco-client-compat = true
```

# 4. 其他配置 #

## 4.1 创建 ocserv 用户 ##

创建 `ocserv` 用户的命令格式是：

```shell
$ sudo ocpasswd -c /etc/ocserv/ocpasswd username
```

例如，要创建一个用户叫 `test`，命令如下：

```shell
$ sudo ocpasswd -c /etc/ocserv/ocpasswd test
```

接着它会要求你输入两次密码，以便确认。

## 4.2 打开 IP 转发 ##

由于 `VPN` 内部需要 `NAT` 功能，所以必须打开 `ipv4` 的转发，设置为如下值：

```shell
$ sudo vim /etc/sysctl.conf

net.ipv4.ip_forward = 1
```

修改保存后，让配置生效，执行下列命令：

```shell
$ sudo /sbin/sysctl -p /etc/sysctl.conf
```

## 4.3 配置 iptables 规则 ##

打开了 `IP` 转发，还需要启用和配置 `NAT`：

```shell
$ sudo iptables -t nat -A POSTROUTING -j MASQUERADE
```

保存 `iptables` 的配置到 `/etc/iptables` 文件：

```shell
$ sudo touch /etc/iptables
$ sudo iptables-save > /etc/iptables
```

在系统启动时恢复 `iptables` 的配置，编辑 `/etc/rc.local` 文件，恢复 `iptables` 的命令必须写在 `exit 0` 语句之前：

```shell
$ sudo vim /etc/rc.local

......... (前面的内容省略)

/sbin/iptables-restore < /etc/iptables
exit 0
```

# 5. 启动 ocserv #

启动的命令如下：

```shell
$ ocserv -c /etc/ocserv/ocserv.conf
```

如果想在系统启动的时候就启动 `ocserv` 服务，可以编辑 `/etc/rc.local` 文件，把上面的启动命令写在 `exit 0` 语句之前：

```shell
$ sudo vim /etc/rc.local

......... (前面的内容省略)

ocserv -c /etc/ocserv/ocserv.conf
exit 0
```

Enjoy it!!

# 6. 参考文章 #

1. [Setup OpenConnect VPN Server for Cisco AnyConnect on Ubuntu 14.04 x64](https://www.vultr.com/docs/setup-openconnect-vpn-server-for-cisco-anyconnect-on-ubuntu-14-04-x64)

2. [在 CentOS 7 上搭建 Cisco AnyConnect VPN](http://blog.csdn.net/y87329396/article/details/48264731)

<.End.>
