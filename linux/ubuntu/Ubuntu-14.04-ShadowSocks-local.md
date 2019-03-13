
# Ubuntu 14.04 下使用 ShadowSocks-local + simple-obfs 上网

## 1. ShadowSocks 客户端

### 1.1 准备工作

#### 1.1.1. 安装 pip

`pip` 是 `python` 的包管理工具，类似于 `Ubuntu` 的 `apt-get`。本文中将使用 `python` 版本的 `ShadowSocks`，因此我们需要通过 `pip` 命令来安装。

在 `Ubuntu 14.04` 下安装 `pip`：

```shell
apt-get update
apt-get install python-pip
```

#### 1.1.2. 更新 pip

```shell
pip install --upgrade pip
```

### 1.2. 安装配置 ShadowSocks

#### 1.2.1. 安装 ShadowSocks

```shell
pip install shadowsocks
```

执行完毕，且没有报错的话，接下来就是配置了。

#### 1.2.2. 配置 ShadowSocks 客户端

新建一个配置文件，例如：`/etc/shadowsocks-local.json`，命令如下：

```shell
vim /etc/shadowsocks-local.json
```

文件内容如下：

```json
{
    "server": "127.0.0.1",
    "server_port": 1984,
    "local_address": "127.0.0.1",
    "local_port": 1080,
    "password": "your_password",
    "timeout": 300,
    "method": "aes-256-cfb",
    "fast_open": false
}
```

#### 1.2.3. 启动 ShadowSocks 客户端

启动命令：

```shell
/usr/local/bin/sslocal -d start -c /etc/shadowsocks-local.json -s 127.0.0.1 -p 1984 -l 1080
或者
nohup /usr/local/bin/sslocal -c /etc/shadowsocks-local.json -s 127.0.0.1 -p 1984 -l 1080 &
```

启动成功后的显示如下所示：

```shell
2019-03-13 17:37:13 WARNING  warning: server set to listen on 127.0.0.1:1984, are you sure?
2019-03-13 17:37:13 INFO     loading libcrypto from libcrypto.so.1.0.0
started
```

#### 1.2.4. 开机自启 service

以下使用 `Systemd` 来实现 `shadowsocks` 开机自启。

如果没有安装 `Systemd`，需要先安装 `Systemd`（可以使用 `systemctl --help` 命令来检查是否安装了 `Systemd`）。

安装的命令是：

```shell
apt-get install systemd
```

新建 `shadowsocks-local` 配置文件：

```shell
vim /etc/systemd/system/shadowsocks-local.service
```

添加如下内容：

```bash
[Unit]
Description=Shadowsocks Client Service
After=network.target

[Service]
Type=simple
User=root
ExecStart=/usr/local/bin/sslocal -c /etc/shadowsocks-local.json -s 127.0.0.1 -p 1984 -l 1080

[Install]
WantedBy=multi-user.target
```

注：因为这里是要通过 `simple-obfs` 混淆后再访问外网，所以这里的 `IP` 必须是本机的内网 `IP` 和端口，一般不用修改。

让 `shadowsocks-local` 配置生效：

```shell
systemctl enable /etc/systemd/system/shadowsocks-local.service
```

执行的结果如下：

```shell
ln -s '/etc/systemd/system/shadowsocks-local.service'
      '/etc/systemd/system/multi-user.target.wants/shadowsocks-local.service'
```

启动 `shadowsocks-local` 服务：

```shell
systemctl start shadowsocks-local
```

查询 `shadowsocks-local` 服务当前的状态：

```shell
systemctl status shadowsocks-local
```

## 2. simple-obfs

### 2.1. simple-obfs 编译安装

`simple-obfs` 的 GitHub 地址：

[https://github.com/shadowsocks/simple-obfs](https://github.com/shadowsocks/simple-obfs)

编译和安装 `simple-obfs`：

```shell
sudo apt-get install --no-install-recommends build-essential autoconf libtool libssl-dev libpcre3-dev libev-dev asciidoc xmlto automake

cd /home
mkdir web
cd web
git clone https://github.com/shadowsocks/simple-obfs.git
cd simple-obfs
git submodule update --init --recursive

./autogen.sh
./configure && make
sudo make install
```

其他 `Linux` 系统的安装方式可以参考 `simple-obfs` 的 GitHub 官网。

### 2.2. simple-obfs 运行

```shell
/usr/local/bin/obfs-local -s your_ss_server_ip -p 8139 -l 1984 --obfs http --obfs-host www.bing.com
```

### 2.3. 开机自启 service

以下使用 `Systemd` 来实现 `shadowsocks` 开机自启。

关于 `Systemd` 的安装请参考第 “`1.2.4.`” 小节，这里不再重复。

新建 `obfs-local` 配置文件：

```shell
vim /etc/systemd/system/obfs-local.service
```

添加如下内容：

```bash
[Unit]
Description=Simple Obfs Client Service
After=network.target

[Service]
Type=simple
User=root
ExecStart=/usr/local/bin/obfs-local -s your_ss_server_ip -p 8139 -l 1984 --obfs http --obfs-host www.bing.com

[Install]
WantedBy=multi-user.target
```

注：请自行修改 `-s your_ss_server_ip -p 8139` 的值，这是你的开启了 `simple-obfs` 服务的 `ShadowSocks` 服务器的 `IP` 和端口设置，否则是不能正常工作的。

让 `obfs-local` 配置生效：

```shell
systemctl enable /etc/systemd/system/obfs-local.service
```

执行的结果如下：

```shell
ln -s '/etc/systemd/system/obfs-local.service'
      '/etc/systemd/system/multi-user.target.wants/obfs-local.service'
```

启动 `obfs-local` 服务：

```shell
systemctl start obfs-local
```

查询 `obfs-local` 服务当前的状态：

```shell
systemctl status obfs-local
```

## 3. socks5 转 http/https

实际使用中，经常会遇到命令行终端或本地程序需要代理，但是他们只支持 `http` 或 `https` 协议，所以就需要把 `socks5` 协议的代理转换协议，这个时候我们需要 `privoxy` 。

### 3.1. 安装 privoxy

安装命令：

```shell
apt-get install privoxy python-m2crypto
```

注意：必须同时安装 `python-m2crypto`，否则在 `Ubuntu 14.04` 上 `privoxy` 可能会运行不起来。

### 3.2. 配置 privoxy

修改配置文件 `/etc/privoxy/config`，

```shell
vim /etc/privoxy/config
```

找到如下两行，这里 `listen-address` 后的端口是未来我们要使用的 `IP` 和端口，默认值为 `8118`。`forward-socks5t` 后的端口是 `ShadowSocks` 使用的本地 `IP` 和端口，这个依据自己的配置修改，不要忘了最后的 “`.`”。（`listen-address` 的值可能默认就不用修改，`forward-socks5t` 的值默认是没有的，需要自己添加，找到关于 `forward-socks` 的相关内容并添加以下的配置，可以在配置文件的 “`5.2. forward-socks4, forward-socks4a, forward-socks5 and forward-socks5t`” 处找到。

```shell
listen-address      localhost:8118
forward-socks5t     /   127.0.0.1:1080    .
```

配置文件的内容有点长，可以使用 `vim` 中的 `/` 命令查找相关关键字。

注：以上配置中的确是 `forward-socks5t`，而不是 `forward-socks5`，也不是笔误。

### 3.3. 启动 privoxy

保存配置后，启动服务：

```shell
service privoxy start
或者
/usr/sbin/privoxy /etc/privoxy/config
```

```shell
export http_proxy=127.0.0.1:8118
export https_proxy=127.0.0.1:8118
export ftp_proxy=127.0.0.1:8118
```

```shell
export http_proxy=
export https_proxy=
export ftp_proxy=
```

```shell
echo $http_proxy
echo $https_proxy
echo $ftp_proxy
```

### 4. genpac

`genpac` 用来获取 `gfwlist` 文件，顺便测试一下 `ShadowSocks` 的 `SOCKS5` 服务是否正常。

安装 `genpac`：

```shell
pip install genpac
pip install --upgrade genpac
```

`pip` 超时的问题，修改超时时间：

```shell
vim ~/.pip/pip.config

[global]
timeout = 60000
index-url = http://e.pypi.python.org/simple
trusted-host = pypi.douban.com

[install]
use-mirrors = true
mirrors = http://e.pypi.python.org
```

以上配置不仅设置了超时时间，同时也添加了下载的镜像地址。

通过镜像网站安装：

```shell
pip install -i http://pypi.douban.com/simple genpac
pip install -i http://pypi.douban.com/simple --upgrade genpac
```

另外推荐另一个站点（清华大学）：[http://e.pypi.python.org](http://e.pypi.python.org)

获取 `gfwlist` 文件，如果下载不了，说明 `SOCKS5` 功能可能不能正常使用。可以尝试把 `--gfwlist-proxy` 参数去掉再试，表示不使用 `SOCKS5` 协议，直接连接。

```shell
genpac --pac-proxy "SOCKS5 127.0.0.1:1080" --gfwlist-proxy "SOCKS5 127.0.0.1:1080" --gfwlist-url "https://raw.githubusercontent.com/gfwlist/gfwlist/master/gfwlist.txt" --output "autoproxy.pac"
```

## X. 参考文章

1. [`Linux 安装配置 ShadowSocks 客户端及开机自动启动`](https://blog.huihut.com/2017/08/25/LinuxInstallConfigShadowsocksClient/)

    [https://blog.huihut.com/2017/08/25/LinuxInstallConfigShadowsocksClient/](https://blog.huihut.com/2017/08/25/LinuxInstallConfigShadowsocksClient/)

2. [`VPS 搭梯子指南——shadowsocks + BBR + obfs`](https://www.solarck.com/shadowsocks-libev.html)

    [https://www.solarck.com/shadowsocks-libev.html](https://www.solarck.com/shadowsocks-libev.html)

3. [`pip 超时(timeout)问题的解决方法`](https://blog.csdn.net/xiaoqu001/article/details/78630392)

    [https://blog.csdn.net/xiaoqu001/article/details/78630392](https://blog.csdn.net/xiaoqu001/article/details/78630392)
