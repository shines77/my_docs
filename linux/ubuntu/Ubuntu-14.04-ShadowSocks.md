Ubuntu 14.04 搭建 Shadowsocks
--------------------------------

在Ubuntu下安装ss很简单。只需要依次执行下面3条命令：

    $ sudo apt-get update
    $ sudo apt-get install python-pip
    $ sudo pip install shadowsocks

pip 是 python 下的方便安装的工具，类似 apt-get。

全部执行完毕且没有报错的话，我们的 shadowsocks 就算是装完了。接下来就是配置部署了。

写一个配置文件保存为 /etc/shadowsocks.json，文件内容如下：

    $ sudo vim /etc/shadowsocks.json

    {
        "server": "my_server_ip",
        "server_port": 8388,
        "local_address": "127.0.0.1",
        "local_port": 1080,
        "password": "mypassword",
        "timeout": 300,
        "method": "aes-256-cfb",
        "fast_open": false
    }

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

server, server_port, password 需要自行根据自己的实际情况修改。

配置文件编辑完成后，接下来就可以部署运行了：

    $ sudo ssserver -c /etc/shadowsocks.json -d start

当然，我们可不希望每次重启服务器都手动启动 SS, 因此我们要把这条命令写到这个文件内(在 exit 0 之前)：/etc/rc.local，这样以后就能开机自动运行了。

    $ sudo vim /etc/rc.local

    if [ $(id -u) -eq 0 ]; then
        /usr/local/bin/ssserver -c /etc/shadowsocks.json -d start
    fi

    exit 0

好了，打开客户端呼吸墙外的空气吧！

