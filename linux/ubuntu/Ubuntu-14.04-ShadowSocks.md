# Ubuntu 14.04 搭建 Shadowsocks 服务器

## 1. 安装和使用

### 1.1. 安装

在 `Ubuntu` 下安装 `Shadowsocks` 很简单。只需要依次执行下面 `3` 条命令：

```shell
sudo apt-get update
sudo apt-get install python-pip
sudo pip install shadowsocks
```

注：`pip` 是 `python` 中安装和更新软件包的工具，类似于 `Ubuntu` 的 `apt-get`。

全部执行完毕，且没有报错的话，我们的 `shadowsocks` 就算是装完了，接下来就是配置了。

### 1.2. 配置

新建一个配置文件，例如：`/etc/shadowsocks.json`，命令如下：

```shell
sudo vim /etc/shadowsocks.json
```

文件内容如下：

```json
{
    "server": "208.110.85.42",
    "server_port": 5173,
    "local_address": "127.0.0.1",
    "local_port": 1080,
    "password": "skyinno251",
    "timeout": 300,
    "method": "aes-256-cfb",
    "fast_open": false
}
```

该配置文件的格式如下：

```json
{
    "server": "你的服务器IP",
    "server_port": 你的服务器端口,
    "local_address": "127.0.0.1",
    "local_port": 1080,
    "password": "你的ShadowSocks密码",
    "timeout": 300,
    "method": "加密格式",
    "fast_open": false
}
```

注：`server`, `server_port`, `password` 请修改为你自己的配置。

### 1.3. 运行

配置文件编辑完成后，接下来就可以部署运行了：

```shell
sudo ssserver -c /etc/shadowsocks.json -d start
```

当然，我们可不希望每次重启服务器都手动启动 `ShadowSocks`，因此我们要把这条命令写到这个文件内 (在 `exit 0` 之前)：`/etc/rc.local`，这样以后就能开机自动运行了。

```shell
sudo vim /etc/rc.local
```

在 `exit 0` 之前添加如下内容：

```bash
if [ $(id -u) -eq 0 ]; then
    /usr/local/bin/ssserver -c /etc/shadowsocks.json -d start
fi

exit 0
```

好了，打开客户端，开始呼吸墙外的空气吧！

## 2. 更新：使用 obfs 混淆

（ `2019年2月14日` 更新）

由于现在 `ShadowSocks` 协议有可能会被主动识别并屏蔽，所以可以采用 `simple-obfs` 或 `GoQuite` 做混淆。具体缘由可看 `ShadowSocks-Windows` 官方 `github` 上的 `issue` 讨论：[有种被针对的感觉](https://github.com/shadowsocks/shadowsocks-windows/issues/2193) 。

`simple-obfs` 官网：[https://github.com/shadowsocks/simple-obfs](https://github.com/shadowsocks/simple-obfs)

`GoQuite` 官网: [https://github.com/cbeuw/GoQuiet/releases](https://github.com/cbeuw/GoQuiet/releases)

本文仅介绍如何使用和配置 `simple-obfs` 来做混淆。

### 2.1. 服务器端

服务器端 (`Ubuntu 14.04 64-bit`)：

关于 `Linux` 上如何编译和安装 `simple-obfs`，可查阅 `simple-obfs` 的官方 `github`，这里不再敖述。

编辑 `rc.local` 文件：

```shell
vim /etc/rc.local
```
如下：

```bash
if [ $(id -u) -eq 0 ]; then
    ulimit -SHn 65535
    /usr/local/bin/obfs-server -s 104.224.132.45 -p 8193 --obfs http -r 127.0.0.1:8388
    /usr/local/bin/ssserver -s 127.0.0.1 -p 8388 -c /etc/shadowsocks.json -d start
fi
```

注：这里的 `ssserver` 由于使用的是本地 `IP` 启动的，会报 `Warning`，在 `/etc/rc.local` 里启动会失败，不能自动启动。所以，待服务器启动以后，需要手动执行一遍启动命令，才能正常启动 `ssserver`，暂时找不到更好的解决办法。命令如下：

```shell
/usr/local/bin/ssserver -s 127.0.0.1 -p 8388 -c /etc/shadowsocks.json -d start`
```

执行完成后，可使用 `top` 命令查看是否已经启动了 `ssserver` 程序。

### 2.2. 客户端

客户端 (`Windows 10`)：

在 `Windows` 客户端上，我们使用 `ShadowSocks-Windows` 的插件方式运行 `simple-obfs`，`simple-obfs` 的 `Windows` 的可执行文件可以在这里下载，[https://github.com/shadowsocks/simple-obfs/releases](https://github.com/shadowsocks/simple-obfs/releases)，并且把 `obfs-local.exe` 和 `libwinpthread-1.dll` 文件放到 `Shadowsocks.exe` 所在的目录，如下图所示：

![simple-obfs 文件目录](./images/ss-obfs-files.png)

鼠标右击 `ShadowSocks-Windows` 的右下角图标，“`服务器`” -> “`编辑服务器...`”，新添或修改你的服务器配置，在 “`插件程序`” 一栏写上 `obfs-local`，在 “`插件选项`” 写上 `obfs=http;obfs-host=www.bing.com` , 配置中的 `http` 要和服务器的设置对应 ( `http` 或 `tls`)，如下图所示：

![编辑服务器](./images/ss-obfs-client.png)

关于 `Windows` 上客户端的设置，更多内容，可参考如下文章：[https://www.jianshu.com/p/135e538164f5](https://www.jianshu.com/p/135e538164f5) 。

## 3. 参考文章

1. [`issue：有种被针对的感觉`](https://github.com/shadowsocks/shadowsocks-windows/issues/2193)

    [https://github.com/shadowsocks/shadowsocks-windows/issues/2193](https://github.com/shadowsocks/shadowsocks-windows/issues/2193)

2. [`ss 客户端使用 obfs 混淆`](https://www.jianshu.com/p/135e538164f5)

    [https://www.jianshu.com/p/135e538164f5](https://www.jianshu.com/p/135e538164f5)

3. [`simple-obfs 官网`](https://github.com/shadowsocks/simple-obfs)

    [https://github.com/shadowsocks/simple-obfs](https://github.com/shadowsocks/simple-obfs)
