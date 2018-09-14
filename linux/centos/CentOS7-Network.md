
CentOS 7.x 网络配置
========================

# 1. 配置命令

想要查询当前的网络配置，可以使用命令：

```shell
$ ip addr

1: lo: <LOOPBACK,UP,LOWER_UP> mtu 65536 qdisc noqueue state UNKNOWN qlen 1
    link/loopback 00:00:00:00:00:00 brd 00:00:00:00:00:00
    inet 127.0.0.1/8 scope host lo
       valid_lft forever preferred_lft forever
    inet6 ::1/128 scope host 
       valid_lft forever preferred_lft forever
2: ens5f0: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc mq state UP qlen 1000
    link/ether 38:d5:47:c7:f0:a9 brd ff:ff:ff:ff:ff:ff
    inet 172.16.66.232/24 brd 172.16.66.255 scope global ens5f0
       valid_lft forever preferred_lft forever
    inet6 fe80::a1e4:33c1:a0e7:adac/64 scope link 
       valid_lft forever preferred_lft forever
3: ens5f1: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc mq state UP qlen 1000
    link/ether 38:d5:47:c7:f0:aa brd ff:ff:ff:ff:ff:ff
    inet 172.16.66.242/24 brd 172.16.66.255 scope global ens5f1
       valid_lft forever preferred_lft forever
    inet6 fe80::b919:8594:24cc:b236/64 scope link 
       valid_lft forever preferred_lft forever
4: ens7f0: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc mq state UP qlen 1000
    link/ether 0c:c4:7a:82:f5:50 brd ff:ff:ff:ff:ff:ff
    inet 172.16.66.243/24 brd 172.16.66.255 scope global ens7f0
       valid_lft forever preferred_lft forever
    inet6 fe80::475f:fa01:c381:9f2f/64 scope link 
       valid_lft forever preferred_lft forever
5: ens7f1: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc mq state UP qlen 1000
    link/ether 0c:c4:7a:82:f5:51 brd ff:ff:ff:ff:ff:ff

...... (后面省略)
```

我们也可以使用下来命令枚举网卡的设备名称：

```shell
$ nmcli connection show

ens5f0  c822baa2-eaff-4b1c-a890-a00f6f49d068  802-3-ethernet  ens5f0 
ens5f1  c6204c53-a666-4b4d-b1a3-7e741a55c8d0  802-3-ethernet  ens5f1 
ens7f0  f3f556a2-f6ff-4c5b-9333-cd7f6cd9195c  802-3-ethernet  ens7f0 
ens7f1  a2b5944f-dcb9-4d10-b454-7f122144aab5  802-3-ethernet  --     
ens7f2  df0a6674-a382-4cc6-8ff0-26e667f68c27  802-3-ethernet  --     
ens7f3  6184c3af-ba6c-4314-86d4-51b51d2e89ae  802-3-ethernet  --
```

那么，如果想修改 `ens5f0` 的网络配置，命令是：

```shell
$ vi /etc/sysconfig/network-scripts/ifcfg-ens5f0
```

修改完成后，重启网络服务：

```shell
$ service network restart
```

