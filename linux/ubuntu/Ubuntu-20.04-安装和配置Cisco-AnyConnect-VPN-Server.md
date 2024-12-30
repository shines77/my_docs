
# Ubuntu 20.04 安装和配置 Cisco AnyConnect VPN Server

## 1. 简介

`Cisco AnyConnect VPN` 是 `Cisco（思科公司）` 的安全远程接入解决方案，隐蔽性要更好一些，穿透性和稳定性也不错，一般不太会掉线。传统的 `VPN` （`LPTP`, `PPTP`, `IPSec` 等方式），以及 `OpenVPN`，受到干扰可能性比较大，比较容易被 `GFW` 探知，如果流量大的话还会被 `GFW` 直接屏蔽。

`Cisco AnyConnect VPN` 还具有下列优势：

* 待机不会自动断开；
* 能够下发路由表给客户端；
* 稳定，隐蔽性好，穿透性强；
* 轻量级，速度快；
* 客户端使用简单；

## 2. 手动编译安装 ocserv

`ocserv` 是一个 `OpenConnect SSL VPN` 协议服务端，从 `0.3.0` 版后开始兼容 `Cisco AnyConnect SSL VPN` 协议的终端。

官方主页：[http://www.infradead.org/ocserv/](http://www.infradead.org/ocserv/)

### 2.1. 下载 ocserv

`Ubuntu 20.04` 提供了 `ocserv` 的安装源，但这里先介绍手动编译安装，使用的版本是最新版 `ocserv 1.1.6` （2022/02/17）。

从官网下载 `ocserv` 的源码包，并解压：

```shell
cd ~/
mkdir ocserv
cd ocserv
wget ftp://ftp.infradead.org/pub/ocserv/ocserv-1.1.6.tar.xz
tar -xf ocserv-1.1.6.tar.xz
cd ocserv-1.1.6
```

### 2.2. 依赖组件

接下来，安装 `ocserv` 的依赖组件：

```bash
sudo apt install build-essential pkg-config libgnutls28-dev libwrap0-dev libpam0g-dev libseccomp-dev libreadline-dev libnl-route-3-dev
```

`ocserv 1.1.6` 相对于 `0.9.2` 等以前的版本，特别增加的依赖组件是 `libev`，安装命令：

```bash
sudo apt install libev-dev
```

### 2.3. 编译安装

开始配置和编译源码，然后安装 `ocserv`：

```shell
sudo ./configure
sudo make
sudo make install
```

## 2. 配置 ocserv

### 3.1. 安装证书工具

先安装一下证书生成工具（`certtool`），并创建一个目录，来存放生成的证书：

```shell
cd ~
sudo apt-get install gnutls-bin
sudo mkdir certificates
cd certificates
```

（其中的 `gnutls-bin` 安装包中包含 `certtool` 工具。）

### 3.2. CA 证书

我们创建一个 `CA` 模板文件（`ca.tmpl`），其中内容如下，你可以修改其中的 `cn` 和 `organization` 字段：

```bash
sudo vim ca.tmpl

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
sudo certtool --generate-privkey --outfile ca-key.pem
sudo certtool --generate-self-signed --load-privkey ca-key.pem --template ca.tmpl --outfile ca-cert.pem
```

### 3.3. Server 证书

接下来，我们创建一个 `server` 证书模板（`server.tmpl`），内容如下。请注意其中的 `cn` 字段，它必须是你的服务器的域名或者 `IP` 地址。

```bash
sudo vim server.tmpl

cn = "Your server's domain name or ip address"
organization = "Your Company Name"
expiration_days = 3650
signing_key
encryption_key
tls_www_server
```

如果你的服务器没有域名，则 `cn` 字段必须使用服务器的 `IP` 地址（这里一定要写正确，否则是连不上的），例如：

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
sudo certtool --generate-privkey --outfile server-key.pem
sudo certtool --generate-certificate --load-privkey server-key.pem --load-ca-certificate ca-cert.pem --load-ca-privkey ca-key.pem --template server.tmpl --outfile server-cert.pem
```

### 3.4. 拷贝证书

新建一个 `/etc/ocserv` 目录，拷贝 `server 证书`, `server key` 和 `ca 证书` 到 `/etc/ocserv` 目录下：

```shell
sudo mkdir -p /etc/ocserv
sudo cp server-cert.pem server-key.pem ca-cert.pem /etc/ocserv
```

### 3.5. 生成客户端证书

跟前面一样，我们创建一个客户端证书模板 “`user.tmpl`”，这里需要先输入一个命令获得当前的时间戳（数值），例如：

```bash
date +%s

1500541096
```

把该命令返回的时间戳 “`1500541096`” 填到 `user.tmpl` 模板的 `serial` 字段里，如下所示：

```bash
sudo vim user.tmpl

cn = "User VPN Client Cert"
unit = "users"
serial = "1500541096"
expiration_days = 3650
signing_key
tls_www_server
```

其中 `cn` 字段是你的客户端证书的用户名，名字可以随意取。注意，这个用户名不是你登录 `Cisco AnyConnect VPN` 的用户名。

然后生成客户端证书 `user-cert.pem`，命令如下：

```bash
sudo certtool --generate-privkey --outfile user-key.pem
sudo certtool --generate-certificate --load-privkey user-key.pem --load-ca-certificate ca-cert.pem --load-ca-privkey ca-key.pem --template user.tmpl --outfile user-cert.pem
```

最后把客户端证书 `user-cert.pem` 转换成 `pkcs12` 格式，生成的客户端证书文件是 `user.p12`，命令如下：

```bash
sudo openssl pkcs12 -export -inkey user-key.pem -in user-cert.pem -name "User VPN Client Cert" -certfile ca-cert.pem -out user.p12
```

其中 `"User VPN Client Cert"` 的名字是可以自定义的，`User` 可以改为 `user.tmpl` 模板里你設置的 `UserName` 的值。

（注：如果你的系统没有安装 `openssl`，需要先安装 `openssl`，以便执行相关命令。）

通过 `http` 服务器或其他方式将客户端证书 `user.p12` 文件传输到客户端，导入即可。

也可以通过 `sz` 命令把文件从 `SSH` 终端下载回来。

先安装 `lrzsz` 工具：

```bash
sudo apt install lrzsz
```

然后使用 `sz` 下载客户端证书文件 `user.p12`：

```bash
sz ~/certificates/user.p12
```

### 3.6. ocserv 配置文件

在 `ocserv` 源代码里有一个简单的配置范例文件：`/ocserv-1.1.6/doc/sample.config`，把它复制到 `/etc/ocserv/` 目录下：

```shell
sudo cp ~/ocserv/ocserv-1.1.6/doc/sample.config /etc/ocserv/config
```

编辑这个配置文件 `/etc/ocserv/config`，找到与下文相同的配置选项，并修改成下文里展示的内容，找到那几个关于 `route` 的配置项，像下面展示的一样，把它们都注释掉。最后一项 “`cisco-client-compat = true`” 在文件比较后面的地方，原本是被注释了的，将其注释去掉。

```bash
sudo nano /etc/ocserv/config

# ocserv 支持多种认证方式，这是自带的密码认证，使用 ocpasswd 创建密码文件
# ocserv 还支持证书认证，可以通过 Pluggable Authentication Modules (PAM) 使用 radius 等认证方式
auth = "plain[/etc/ocserv/ocpasswd]"

# tcp 和 udp 端口，默认值是 443，可以不用改
tcp-port = 443
udp-port = 443

# 运行用户和组，默认值是 nobody 和 daemon
run-as-user = ocserv
run-as-group = ocserv

# 服务器证书路径
server-cert = /etc/ocserv/server-cert.pem
server-key = /etc/ocserv/server-key.pem

# 客户端证书路径
ca-cert = /etc/ocserv/ca-cert.pem

# 最大客户端连接数，默认值是 16
max-clients = 32

# 同一个用户名最大同时登陆连接数，默认值是 2
max-same-clients = 8

# 默认是 false, 修改为 true
try-mtu-discovery = true

# 下面这个选项从 1.1.2 开始就取消了，需要注释掉, 如果没有则略过
# listen-clear-file = /var/run/ocserv-conn.socket

# TLS 的优先顺序
# tls-priorities = "NORMAL:%SERVER_PRECEDENCE:%COMPAT:-VERS-SSL3.0"
# tls-priorities = "NORMAL:%SERVER_PRECEDENCE:%COMPAT:-VERS-SSL3.0:-VERS-TLS1.0:-VERS-TLS1.1"
tls-priorities = "NORMAL:%SERVER_PRECEDENCE:%COMPAT:-VERS-SSL3.0:-VERS-TLS1.0:-VERS-TLS1.1:-VERS-TLS1.2"

# 最小验证重试时间：单位(秒)
min-reauth-time = 300

# 要公布的默认域，如果没有域名，可以改为你的服务器的 IP 地址
default-domain = example.com

# VPN 客户端的网络(IPv4)，IP 池范围，要跟你的 VPN 客户端本地的局域网网段错开
# 这里选一个比较少人用的网段 10.250.x.x
# 默认值为:
# ipv4-network = 192.168.1.0
# ipv4-netmask = 255.255.255.0
ipv4-network = 10.250.0.0
ipv4-netmask = 255.255.0.0

# DNS 设置
dns = 8.8.8.8
dns = 8.8.4.4

# 香港本地的 DNS (备用)
# dns = 202.238.95.24
# dns = 202.238.95.26

# 请参考下面，注释掉所有的 route, noroute 的定义，这样的效果是 VPN 客户端所有的访问都通过 VPN 代理转发。
# 其中 route 字段表示使用 VPN 代理转发的网段，noroute 字段表示不使用 VPN 代理转发的网段。
# 注：最多仅支持 60 条 route 规则或 60 条 noroute 规则。这些路由规则是下发到 VPN 客户端的。

# 合理的配置路由可以往国内的网站不走 VPN，国外的网站走 VPN，但不会设置尽量别乱设置。

# 如果不会配置，请把 route 都注释掉。
#route = 10.10.10.0/255.255.255.0
#route = 192.168.0.0/255.255.0.0
#route = fef4:db8:1000:1001::/64

# 这里表示不使用 VPN 代理转发的网段，下面是所有的私有地址段。
no-route = 127.0.0.0/255.0.0.0
no-route = 169.254.0.0/255.255.0.0
no-route = 172.16.0.0/255.240.0.0
no-route = 192.168.0.0/255.255.0.0

# 你也可以把 VPN 服务器 IP 所在的网段也加到 no-route 里，例如：114.128.111.222
no-route = 10.3.0.0/255.255.0.0
no-route = 114.128.111.0/255.255.255.0

# 以下选项适用于 AnyConnect 客户端兼容性（实验性）。

#
# The following options are for (experimental) AnyConnect client
# compatibility.
#

# This option will enable the pre-draft-DTLS version of DTLS, and
# will not require clients to present their certificate on every TLS
# connection. It must be set to true to support legacy CISCO clients
# and openconnect clients < 7.08. When set to true, it implies dtls-legacy = true.
cisco-client-compat = true

# This option allows one to disable the DTLS-PSK negotiation (enabled by default).
# The DTLS-PSK negotiation was introduced in ocserv 0.11.5 to deprecate
# the pre-draft-DTLS negotiation inherited from AnyConnect. It allows the
# DTLS channel to negotiate its ciphers and the DTLS protocol version.
#dtls-psk = false

# This option allows one to disable the legacy DTLS negotiation (enabled by default,
# but that may change in the future).
# The legacy DTLS uses a pre-draft version of the DTLS protocol and was
# from AnyConnect protocol. It has several limitations, that are addressed
# by the dtls-psk protocol supported by openconnect 7.08+.
dtls-legacy = true

# This option will enable the X-CSTP-Client-Bypass-Protocol (disabled by default).
# If the server has not configured an IPv6 or IPv4 address pool, enabling this option
# will instruct the client to bypass the server for that IP protocol. The option is
# currently only understood by Anyconnect clients.
client-bypass-protocol = false

# 把最后的 [vhost:www.example.com] 的默认范例设置内容都删掉, 如下所示:

# An example virtual host with different authentication methods serviced
# by this server.

[vhost:www.example.com]
......
xxxxxx
xxxxxx
xxxxxx
(省略...)
......
```

附，香港本地可用的的 `DNS`：

```text
港服：          202.238.95.24、202.238.95.26
香港特别行政区： 202.181.224.2、203.80.96.10、202.45.84.58
香港 BBN：      203.80.96.10
```

由于我们的配置文件里指定了 `ocserv` 用户和 `ocserv` 组，所以我们要添加这个用户和组，命令如下：

```shell
sudo groupadd -f -r -g 21 ocserv
sudo useradd -M -s /sbin/nologin -g ocserv ocserv
sudo passwd ocserv
sudo gpasswd -a ocserv sudo
```

其中第三步的时候会让你输入两遍密码，该密码不是很重要，但由于是系统用户，最好记得密码，且密码不能过于简单。

## 4. 其他配置

### 4.1. 创建 ocserv 用户

创建 `ocserv` 用户的命令格式是：

```shell
sudo ocpasswd -c /etc/ocserv/ocpasswd username
```

例如，要创建一个用户叫 `test`，命令如下：

```shell
sudo ocpasswd -c /etc/ocserv/ocpasswd test
```

接着它会要求你输入两次密码，以便确认。

### 4.2. 打开 IP 转发

由于 `VPN` 内部需要 `NAT` 功能，所以必须打开 `ipv4` 的转发，设置为如下值：

```shell
sudo vim /etc/sysctl.conf

net.ipv4.ip_forward = 1
```

修改保存后，让配置生效，执行下列命令：

```shell
sudo /sbin/sysctl -p /etc/sysctl.conf
```

### 4.3. 配置 iptables 规则

打开了 `IP` 转发，还需要启用和配置 `NAT`：

```shell
sudo iptables -t nat -A POSTROUTING -j MASQUERADE
```

可以通过下面的命令查看 `NAT` 的设置：

```shell
sudo iptables -t nat --list

Chain PREROUTING (policy ACCEPT)
target     prot opt source                destination

Chain INPUT (policy ACCEPT)
target     prot opt source                destination

Chain OUTPUT (policy ACCEPT)
target     prot opt source                destination

Chain POSTROUTING (policy ACCEPT)
target     prot opt source                destination
MASQUERADE  all  --  anywhere             anywhere
```

其中最后一条就是我们刚才添加的路由规则。

由于路由配置重启了会失效，所以我们要把它保存到一个文件里，然后在重启的时候，再从这个文件来恢复路由配置。

下面的命令即是保存 `iptables` 的配置到 `/etc/iptables` 文件里：

```shell
sudo touch /etc/iptables
sudo iptables-save > /etc/iptables
```

在系统启动时，恢复 `iptables` 的配置，编辑 `/etc/rc.local` 文件，恢复 `iptables` 的命令必须写在 `exit 0` 语句之前：

```shell
sudo vim /etc/rc.local

......... (前面的内容省略)

/sbin/iptables-restore < /etc/iptables
```

## 5. 启动 ocserv

启动的命令如下：

```shell
sudo /usr/local/sbin/ocserv -c /etc/ocserv/config -f -d 1
```

如果想在系统启动的时候就启动 `ocserv` 服务，可以编辑 `/etc/rc.local` 文件，把上面的启动命令写在 `exit 0` 语句之前（但是要记得去掉 `-f` 参数，否则你的系统重启时会卡在启动 `ocserv`，无法完全进入系统，切记！）：

```shell
sudo vim /etc/rc.local

......... (前面的内容省略)

/usr/local/sbin/ocserv -c /etc/ocserv/config -d 1
```

## 6. 开机自启 ocserv

上面的启动方式还是不够完美，我们可以使用 `Systemd` 来实现 `ocserv` 的开机自启。

先通过 `systemctl --help` 命令来检查是否已经安装了 `Systemd`，如果没有安装，则使用如下命令安装：

```shell
sudo apt-get install systemd
```

新建 `ocserv` 配置文件：

```shell
sudo vim /etc/systemd/system/ocserv.service
```

添加如下内容：

```bash
[Unit]
Description=Cisco VPN Server Service
After=network.target

[Service]
Type=simple
User=root
ExecStart=/usr/local/sbin/ocserv -c /etc/ocserv/config -f -d 1

[Install]
WantedBy=multi-user.target
```

让 `ocserv` 配置生效：

```shell
sudo systemctl enable /etc/systemd/system/ocserv.service
```

执行的结果如下：

```shell
ln -s '/etc/systemd/systemocserv.service'
      '/etc/systemd/system/multi-user.target.wants/ocserv.service'
```

启动 `ocserv` 服务：

```shell
sudo systemctl start ocserv
```

查询 `ocserv` 服务当前的状态：

```shell
sudo systemctl status ocserv
```

如果显示报错信息，则代表启动失败，请检查设置文件 `/etc/ocserv/config` 。

确认已经开启成功：

```shell
netstat -tulpn | grep 443
```

```shell
tcp      0     0 0.0.0.0:443     0.0.0.0:*     LISTEN      511/ocserv-main
tcp6     0     0 :::443          :::*          LISTEN      511/ocserv-main
udp      0     0 0.0.0.0:443     0.0.0.0:*                 511/ocserv-main
udp6     0     0 :::443          :::*                      511/ocserv-main
```

搞定，Enjoy it now !!

## 7. 参考文章

1. [Setup OpenConnect VPN Server for Cisco AnyConnect on Ubuntu 14.04 x64](https://www.vultr.com/docs/setup-openconnect-vpn-server-for-cisco-anyconnect-on-ubuntu-14-04-x64)

2. [在 CentOS 7 上搭建 Cisco AnyConnect VPN](http://blog.csdn.net/y87329396/article/details/48264731)

3. [ocserv 1.1.6 - manual.html](http://ocserv.gitlab.io/www/manual.html)

<.End.>
