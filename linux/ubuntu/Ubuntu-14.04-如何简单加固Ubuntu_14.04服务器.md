
# 如何简单加固 Ubuntu 14.04 服务器

## 1. 系统版本

`Ubuntu 14.04.x LTS` 并开启 `SSH` 服务。

## 2. 更新系统安全补丁

保持操作系统的更新是安装好任何操作系统之后的一个必要步骤，这会减少当前操作系统中的已知漏洞。`Ubuntu 14.04` 或其它 `Ubuntu` 版本都可以使用如下命令进行更新：

```shell
sudo apt-get update
sudo apt-get upgrade
sudo apt-get autoremove
sudo apt-get autoclean
```

## 3. 开启安全补丁自动更新

对安全补丁启用自动更新是非常有必要的，这在很大程度上能够保证我们的服务器安全，要启用 `Ubuntu 14.04.x` 的自动更新功能必需先使用如下命令安装无从应答模块：

```shell
sudo apt-get install unattended-upgrades
```

再使用如下命令进行启用：

```shell
sudo dpkg-reconfigure -plow unattended-upgrades
```

执行上述命令后，会自动创建一个 `/etc/apt/apt.conf.d/20auto-upgrades` 文件，并写入如下配置内容：

```
APT::Periodic::Update-Package-Lists "1";
APT::Periodic::Unattended-Upgrade "1";
```

不需要的时候我们可将配置文件删除或清空即可。

## 4. 禁用 root 账户

由于 `root` 账户权力过大，我们可以使用如下命令将其禁用：

```shell
sudo passwd -l root
```

如果想重新启用 `root` 账户可以使用如下命令：

```shell
sudo passwd -u root
```

## 5. 禁用 IPv6

由于 `IPv6` 还并没有在全球范围普及开来，这会造成连接缓慢的问题，我们可以将 `IPv6` 功能暂时禁用掉。要禁用 `IPv6`，我们可以编辑如下配置文件：

```shell
sudo vim /etc/sysctl.conf
```

将 `IPv6` 的相关配置文件改成如下样子：

```
net.ipv6.conf.all.disable_ipv6 = 1
net.ipv6.conf.default.disable_ipv6 = 1
net.ipv6.conf.lo.disable_ipv6 = 1
```

更改完成后，需要执行如下命令使其生效：

```shell
sudo sysctl –p
```

## 6. 禁用 RQBALANCE 特性

`RQBALANCE` 特性主要用于在多个 `CPU` 间分发硬件中断来提高性能，我建议禁用 `RQBALANCE` 特性以避免线程被硬件中断。

要禁用 `RQBALANCE` 特性需要编辑如下配置文件：

```shell
sudo vim /etc/default/irqbalance
```

将 `ENABLED` 的值改为 `0`：

```
ENABLED=0
```

## 7. 修复 OpenSSL 心血漏洞

`Heartbleed` 漏洞闹得沸沸扬扬，在这里我就不介绍了，大家可以自己 `Google`。我需要说明的是：心血漏洞出现在 `OpenSSL` 的如下版本中：

```
1.0.1
1.0.1a
1.0.1b
1.0.1c
1.0.1d
1.0.1e
1.0.1f
```

如果你在使用上述版本就得尽快升级了，查看 `OpenSSL` 版本的命令如下：

```shell
sudo openssl version -v
sudo openssl version -b
```

如果返回的版本是 `1.0.1`，或者时间早于 `2014` 年 `4` 月 `7` 日的版本，就有可能遭受 `Heartbleed` 攻击，得尽快更新才行。

## 8. 参考文章

* [如何简单加固 Ubuntu 14.04 服务器](https://www.sysgeek.cn/hardening-ubuntu-server-14-0-4/)

    [https://www.sysgeek.cn/hardening-ubuntu-server-14-0-4/](https://www.sysgeek.cn/hardening-ubuntu-server-14-0-4/)

<.End.>
