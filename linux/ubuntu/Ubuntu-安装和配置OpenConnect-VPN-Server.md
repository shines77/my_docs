# Ubuntu 安装和配置 OpenConnect-VPN-Server

## 1. 简介

`Cisco AnyConnect VPN` 已更名为 `OpenConnect-VPN-Server`，但是可执行文件还是叫 `ocserv`。

新的官网是：[https://ocserv.openconnect-vpn.net/](https://ocserv.openconnect-vpn.net/)

旧的官网是：[http://www.infradead.org/ocserv/](http://www.infradead.org/ocserv/)

## 2. 源安装

Ubuntu / Debian:

```bash
apt-get install -y ocserv
```

RHEL / CentOS / Fedora:

```bash
yum install -y ocserv
```

## 2. 编译安装

### 2.1. 下载 ocserv

最新的版本是： `ocserv 1.5.0` （2026-06-07 发布）。

从官网下载 `ocserv` 的源码包，并解压：

``````bash
cd ~/
mkdir ocserv
cd ocserv

# FTP 的下载地址还可以用, 但不推荐
# wget ftp://ftp.infradead.org/pub/ocserv/ocserv-1.5.0.tar.xz

wget https://www.infradead.org/ocserv/download/ocserv-1.5.0.tar.xz
tar -xf ocserv-1.5.0.tar.xz
cd ocserv-1.5.0
```

### 2.2. 依赖组件

接下来，安装 `ocserv` 的依赖组件：

```bash
sudo apt install build-essential pkg-config libgnutls28-dev libwrap0-dev libpam0g-dev libseccomp-dev libreadline-dev libnl-route-3-dev
```

`ocserv 1.5.0` 相对于 `0.9.2` 等以前的版本，特别增加的依赖组件是 `libev`，安装命令：

```bash
sudo apt install libev-dev
```

### 2.3. 编译安装

开始配置和编译源码，然后安装 `ocserv`：

```bash
sudo ./configure
sudo make
sudo make install
```

## 3. 生成证书

### 3.1. 安装证书工具

如果 Ubuntu 已经有了 `certtool` 工具，则可跳过此步。

如果没有，则先安装一下：

```bash
sudo apt-get install gnutls-bin
```

（其中的 `gnutls-bin` 安装包中包含 `certtool` 工具。）

### 3.2 CA 和 Server 证书模板

然后创建 CA 证书模板和 Server 证书模板文件。

创建一个目录，来存放生成的证书：

```bash
sudo mkdir /root/certificates
cd /root/certificates
```

请根据此示例文件创建 CA 和 Server 证书模板。

```bash
sudo vim ca.tmpl

cn = "Your organization’s certificate authority"  
organization = "Your Company Name or Organization"  
serial = 1  
expiration_days = 3650  
ca  
signing_key  
cert_signing_key  
crl_signing_key
```

根据您的组织名称和需求来配置相关参数。请注意，任何使用 anyconnect 的 VPN 客户端在证书与主机名不匹配或证书为自签名时都会产生错误提示。

例如：

```bash
cn = "VPN CA"
organization = "Cisco Inc."
serial = 1
expiration_days = 3650
ca
signing_key
cert_signing_key
crl_signing_key
```

接下来，我们创建一个 `server` 证书模板（`server.tmpl`），根据您的组织名称和需求来修改相关参数，内容如下。

```bash
sudo vim server.tmpl

cn = "Your server's domain name or ip address, usually matches hostname"
organization = "Your Company Name or Organization"
expiration_days = 3650
signing_key
encryption_key
tls_www_server
dns_name = "Your organization's hostname"
# ip_address = "If no hostname uncomment and set the IP address here"
```

如果你的服务器没有域名，则可以把 `ip_address` 的注释去掉，并填入你的 IP 地址。也可以直接把 IP 地址写到 `cn` 里。

例如：

```bash
cn = "www.example.com"
organization = "Cisco Inc."
expiration_days = 3650
signing_key
encryption_key
tls_www_server
dns_name = "www.example.com"
```

（注：这里的 `organization` 字段可以跟 `ca.tmpl` 里的 `organization` 名字不一样。）

### 3.3 生成 CA 密钥和 CA 证书

生成一个随机的密钥 `CA key` ，并用这个密钥和 `ca.tmpl` 模板生成 `CA 证书`（`ca-cert.pem`）。

```bash
sudo certtool --generate-privkey --outfile ca-key.pem
sudo certtool --generate-self-signed --load-privkey ca-key.pem --template ca.tmpl --outfile ca-cert.pem
```

### 3.4 生成 Server 密钥和证书

生成一个随机的密钥 `Server key`，并使用这个 `Server key` 密钥、`CA key` 密钥、`CA 证书` 以及 `server.tmpl` 模板生成 `server 证书`（`server-cert.pem`）。

```bash
sudo certtool --generate-privkey --outfile server-key.pem
sudo certtool --generate-certificate --load-privkey server-key.pem --load-ca-certificate ca-cert.pem --load-ca-privkey ca-key.pem --template server.tmpl --outfile server-cert.pem
```

### 3.5 拷贝证书

新建一个 `/etc/ocserv` 目录，把 `server 证书`, `server key` 密钥和 `ca 证书` 拷贝到该目录下：

```bash
sudo mkdir -p /etc/ocserv
sudo cp server-cert.pem server-key.pem ca-cert.pem /etc/ocserv
```

## 4. 配置 ocserv

如果是用源码安装的，则需要自己从 `ocserv` 源代码里复制配置范例文件到 `/etc/ocserv/` 目录下：

```bash
sudo cp ~/ocserv/ocserv-1.5.0/doc/sample.config /etc/ocserv/ocserv.conf
```

如果是从源安装的，默认已经有一个配置文件了，直接编辑该配置文件。

```bash
sudo vim /etc/ocserv/ocserv.conf

# ocserv 支持多种认证方式，这是自带的密码认证，使用 ocpasswd 创建密码文件
# ocserv 还支持证书认证，可以通过 Pluggable Authentication Modules (PAM) 使用 radius 等认证方式
# auth = "pam"
auth = "plain[/etc/ocserv/ocpasswd]"

# tcp 和 udp 端口，默认值是 443，可以不用改
tcp-port = 443
udp-port = 443

# 运行用户和组，默认值是 nobody 和 daemon
run-as-user = ocserv
run-as-group = ocserv

# socket file used for IPC with occtl. You only need to set that,
# if you use more than a single servers.
#occtl-socket-file = /run/occtl.socket

# socket file used for server IPC (worker-main), will be appended with .PID
# It must be accessible within the chroot environment (if any), so it is best
# specified relatively to the chroot directory.
socket-file = /run/ocserv-socket

# The default server directory. Does not require any devices present.
chroot-dir = /var/lib/ocserv

# 服务器证书路径
server-cert = /etc/ocserv/server-cert.pem
server-key = /etc/ocserv/server-key.pem

# 客户端证书路径
ca-cert = /etc/ocserv/ca-cert.pem

# 在 seccomp 章节，需要决定是否使用 seccomp。如果你在编译时删除了 seccomp，
# 或者没有安装 seccomp 相关软件，那么禁用 seccomp 或 ocserv 将会导致程序无法启动。
isolate-workers = true

# 最大客户端连接数，默认值是 16
max-clients = 32

# 同一个用户名最大同时登陆连接数，默认值是 2
max-same-clients = 4

# 默认是 false, 修改为 true
try-mtu-discovery = true

# 下面这个选项从 1.1.2 开始就取消了，需要注释掉, 如果没有则略过
# listen-clear-file = /var/run/ocserv-conn.socket

# TLS 的优先顺序
# tls-priorities = "NORMAL:%SERVER_PRECEDENCE:%COMPAT:-VERS-SSL3.0"
# tls-priorities = "NORMAL:%SERVER_PRECEDENCE:%COMPAT:-VERS-SSL3.0:-VERS-TLS1.0:-VERS-TLS1.1"
# tls-priorities = "NORMAL:%SERVER_PRECEDENCE:%COMPAT:-VERS-SSL3.0:-VERS-TLS1.0:-VERS-TLS1.1:-VERS-TLS1.2"
tls-priorities = "NORMAL:%SERVER_PRECEDENCE:%COMPAT:-VERS-SSL3.0:-VERS-TLS1.0:-VERS-TLS1.1:-VERS-TLS1.3"

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
dns = 8.8.8.4

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
no-route = 10.250.0.0/255.255.0.0
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

保存文件然后退出（使用 CTRL + O 进行保存，CTRL + X 用于退出）。

由于我们的配置文件里指定了 `ocserv` 用户和 `ocserv` 用户组，所以我们要添加这个用户和用户组，命令如下：

```bash
sudo groupadd -f -r -g 21 ocserv
sudo useradd -M -s /sbin/nologin -g ocserv ocserv
sudo passwd ocserv
sudo gpasswd -a ocserv sudo
```

其中第三步的时候会让你输入两遍密码，该密码不是很重要，但由于是系统用户，最好记得密码，且密码不能过于简单。

## 5. 启动 ocserv

```bash
vim /lib/systemd/system/ocserv.service
```

如果你是用源安装的，直接执行下面的命令：

```bash
systemctl start ocserv.service
```

设置为开机启动：

```bash
systemctl enable ocserv.service
```

检查运行状态：

```bash
systemctl status ocserv.service
```

## 6. 参考

1. [Ocserv Configuration - Basic](https://docs.openconnect-vpn.net/recipes/ocserv-configuration-basic/)
