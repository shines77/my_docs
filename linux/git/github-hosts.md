# Linux 下修改 hosts 加速访问 Github

## 1. Hosts

编辑 Hosts 文件：

```shell
sudo vim /etc/hosts
```

插入以下内容：

```shell
# Github
151.101.44.249 github.global.ssl.fastly.net
# 192.30.253.113 github.com
140.82.114.3 github.com
199.232.28.133 raw.githubusercontent.com
103.245.222.133 assets-cdn.github.com
23.235.47.133 assets-cdn.github.com
203.208.39.104 assets-cdn.github.com
204.232.175.78 documentcloud.github.com
204.232.175.94 gist.github.com
107.21.116.220 help.github.com
207.97.227.252 nodeload.github.com
199.27.76.130 raw.github.com
107.22.3.110 status.github.com
204.232.175.78 training.github.com
# 207.97.227.243 www.github.com
140.82.114.3 www.github.com
185.31.16.184 github.global.ssl.fastly.net
185.31.18.133 avatars0.githubusercontent.com
185.31.19.133 avatars1.githubusercontent.com
```

重置网络配置：

Ubuntu 18.04 以下，或者 Ubuntu 18.04 桌面版，使用如下命令:

```shell
sudo /etc/init.d/networking restart
```

Ubuntu 18.04 Server 版，使用如下命令：

```shell
netplan apply
```

## 2. 参考文章

* [Linux下，修改hosts，加速访问Github](https://bmvps.com/github-202005/)
* [HOSTS大法解决Github Clone太慢](cnblogs.com/ocean1100/p/9442962.html)
