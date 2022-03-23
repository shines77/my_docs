# 如何在 Linux 终端安装和使用 Cisco AnyConnect VPN Client

## 1. 安装 Cisco AnyConnect Client

参考文章：[Install Cisco AnyConnect on Ubuntu / Debian / Fedora](https://computingforgeeks.com/install-cisco-anyconnect-on-ubuntu-debian-fedora/) [需要翻墙]

### 1.1 下载 Cisco AnyConnect Client

先到 `Cisco` 的官网下载 `Linux` 版的 `AnyConnect`，官网下载页面是这：[https://software.cisco.com/download/home](https://software.cisco.com/download/home)，进入以后，输入 “AntConnect”，自动完成菜单可能会弹出如下信息：

```shell
AnyConnect Secure Mobility Client
AnyConnect Secure Mobility Client v4.x
```

如果网络太卡，没有弹出自动完成信息，也可以直接按回车，跳转后看到的内容是一样的，点击任何一个均可，最终进入 “AnyConnect Secure Mobility Client v4.x” 的页面里，不要被 "`Mobility`" 迷惑了，这里不是 "`移动端`" 的意思。

在下载列表里找到 “AnyConnect Pre-Deployment Package (Linux 64-bit)”，下载的文件名类似为 “anyconnect-linux64-4.10.05085-predeploy-k9.tar.gz”。但是由于 `Cisco` 需要你注册账号并获得经销商的合同才能下载文件，所以我们只能从别的地方找下载文件。

我在这里找到了一个版本：[Install and Configure the Cisco AnyConnect Software VPN on Linux](https://uci.service-now.com/kb_view.do?sysparm_article=KB0010201)，文章的开头有下载链接，下载地址是这个：
[Anyconnect VPN client](https://uci.service-now.com/sys_attachment.do?sys_id=1507c0cc1b200dd44d61baeedc4bcbe9)，这里版本可能会不断更新，我下载到的版本是：`4.10.04065`。

### 1.2 安装 Cisco AnyConnect Client

下载到安装包以后，可以通过 `SFTP` 软件把文件传到服务器上，也可以使用 `Linux` 的 `sz`, `rz` 命令，通过 `SSH` 终端上传文件。

```shell
# 先安装 sz, rz 包
# Ubuntu / Debian
sudo apt-get install lszrz
# CentOS
sudo yum install lszrz
```

注：`Ubuntu 20.04` 里没有 `lszrz` 安装包，但默认已经自带了 `sz`, `rz`。

使用下列命令上传文件：

```shell
rz
```

输入 rz 命令之后，`SSH 终端` 会让你选择本地的文件，如果你的 `SSH 终端` 不支持，可以换成 "`X Shell 5.0`" 之类的。另外需要注意的一点是，`sz`, `rz` 命令不能在已经用 `screen` 命令登陆服务器的情况，会卡住无法上传，过一会才能恢复。

安装 Cisco AnyConnect Client：

```shell
sudo apt-get update
tar -zxvf anyconnect-linux64-4.10.04065-predeploy-k9.tar.gz
cd anyconnect-linux64-4.10.04065/vpn
sudo ./vpn_install.sh
```

## 3. 连接 VPN Server

### 3.1 用户名和密码

先创建一个用户名和密码的配置文件：

```shell
cd /etc/init
sudo vim cisco_vpn_creds.conf
```

把配置文件放在系统文件夹 "`/etc/init`" 里 (非 Ubuntu 系统可能路径不同，请自行更改)。如果没有 `root` 权限，也可以把文件保存在你当前登录用户的根目录下，例如路径："`~/.cisco_vpn_creds.conf`"。

配置文件的格式为：

```shell
vpn_server_username
vpn_server_password
y
```

例如：

```shell
Tommy
abc12345
y
```

### 3.2 启动脚本

把启动脚本放在系统固定的脚本路径 "`/etc/init.d`" 里 (非 Ubuntu 系统可能路径不同，请自行更改)，放在系统目录的好处是，不用输入路径前缀，任何地方直接输入启动脚本的文件名即可启动。同理，也可以把它保存在你的用户目录的根目录下，但是启动的时候需要加上相应的路径。

```shell
cd /etc/init.d
sudo vim cisco_vpn_connect.sh
```

内容如下：

```shell
#!/bin/bash

if [[ -n "${1}" ]]; then
    VPN_SERVER_HOST="${1}"
else
    VPN_SERVER_HOST="184.170.218.90:10443"
fi

if [[ -n "${2}" ]]; then
    VPN_CREDS_FILE="${2}"
else
    VPN_CREDS_FIL="/etc/init/cisco_vpn_creds.conf"
fi

echo "Connecting to VPN ..."

/opt/cisco/anyconnect/bin/vpn -s < ${VPN_CREDS_FILE} connect ${VPN_SERVER_HOST}
```

设置执行权限：

```shell
sudo chmod +x cisco_vpn_connect.sh
```

启动脚本的命令行格式为：

```shell
cisco_vpn_connect.sh ${VPN_SERVER_HOST} ${VPN_CREDS_FILE}
```

如果省略两个参数，则使用脚本内默认的服务器 `IP` 和配置文件，也可以指定一个或两个参数。

### 3.3 断开脚本

```shell
cd /etc/init.d
sudo vim cisco_vpn_disconnect.sh
```

内容如下：

```shell
#!/bin/bash

if [[ -n "${1}" ]]; then
    VPN_SERVER_HOST="${1}"
else
    VPN_SERVER_HOST="184.170.218.90:10443"
fi

echo "Disconnecting VPN ..."

/opt/cisco/anyconnect/bin/vpn -s disconnect ${VPN_SERVER_HOST}
```

设置执行权限：

```shell
sudo chmod +x cisco_vpn_disconnect.sh
```

命令行的格式为：

```shell
cisco_vpn_disconnect.sh ${VPN_SERVER_HOST}
```

注意：断开连接的脚本指定的 "`VPN服务器IP`" 必须跟启动连接的时的 `IP` 保持一致。

### 3.4 查看连接状态

```shell
/opt/cisco/anyconnect/bin/vpn state
```

显示的信息如下：

```shell
Cisco AnyConnect Secure Mobility Client (version 4.7.01076) .

Copyright (c) 2004 - 2019 Cisco Systems, Inc.  All Rights Reserved.


  >> state: Connected
  >> state: Connected
  >> state: Connected
  >> registered with local VPN subsystem.
VPN>
```

## 4. 参考文章

* [Install Cisco AnyConnect on Ubuntu / Debian / Fedora](https://computingforgeeks.com/install-cisco-anyconnect-on-ubuntu-debian-fedora/) [需要翻墙]

* [Connect To VPN Server with Cisco AnyConnect from Linux Terminal](https://computingforgeeks.com/connect-to-vpn-server-with-cisco-anyconnect-from-linux-terminal/) [需要翻墙]

* [Install and Configure the Cisco AnyConnect Software VPN on Linux](https://uci.service-now.com/kb_view.do?sysparm_article=KB0010201)
