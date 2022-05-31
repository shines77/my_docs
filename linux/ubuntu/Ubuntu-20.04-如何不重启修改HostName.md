# 如何在 Ubuntu 20.04 不重启修改 HostName

## 1. 前言

主机名是在操作系统的安装过程中设置的，或者在创建虚拟机时动态分配给虚拟机的。本指南说明了如何在 `Ubuntu 20.04` 上设置或更改主机名，而无需重新启动系统。

## 2. 了解主机名(hostname)

主机名(hostname) 是标识网络上设备的标签。同一网络上不应有两台或更多台具有相同主机名 (hostname) 的计算机。

在 `Ubuntu` 中，您可以使用以下 `hostnamectl` 命令编辑系统主机名(hostname)和相关设置。

该工具可识别三种不同的主机名(hostname)类别：

* `static` - 传统主机名(hostname)。它存储在/etc/hostname文件中，可以由用户设置。
* `pretty` - 用于向用户展示的描述性自由格式UTF8主机名(hostname)。例如，Linuxize's laptop。
* `transient` - 由内核维护的动态主机名(hostname)。DHCP或mDNS服务器可以在运行时更改临时主机名(hostname)。默认情况下，它与static主机名(hostname)相同。

建议使用完全合格的域名（`FQDN`），如 `host.example.com` 两个 `static` 和 `transient` 名称。

只有 root 或具有 sudo 特权的用户才能更改系统主机名(hostname)。

## 3. 显示当前主机名(hostname)

要查看当前主机名(hostname)，请在hostnamectl不使用任何参数的情况下调用命令：

```shell
hostnamectl
```

返回信息：

```text
   Static hostname: Ubuntu2020.localdomain
         Icon name: computer-vm
           Chassis: vm
        Machine ID: XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
           Boot ID: YYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYY
    Virtualization: kvm
  Operating System: Ubuntu 20.04.4 LTS
            Kernel: Linux 5.4.0-47-generic
      Architecture: x86-64
```

在此示例中，当前主机名(hostname) 设置为 `Ubuntu2020.localdomain`。

## 4. 更改系统主机名(hostname)

更改系统主机名(hostname)是一个简单的过程。语法如下：

```shell
sudo hostnamectl set-hostname host.example.com
sudo hostnamectl set-hostname "Your Pretty HostName" --pretty
sudo hostnamectl set-hostname host.example.com --static
sudo hostnamectl set-hostname host.example.com --transient
```

例如，要将系统静态主机名(hostname)更改为 `Ubuntu-20`，可以使用以下命令：

```shell
sudo hostnamectl set-hostname Ubuntu-20
```

您也可以选择设置漂亮的主机名(hostname)：

```shell
sudo hostnamectl set-hostname "shines77's laptop" --pretty
```

`hostnamectl` 不产生输出。成功时，返回 0，否则返回非零失败代码。

静态主机名(hostname) 存储在中 `/etc/hostname`，漂亮主机名(hostname)存储在 `/etc/machine-info` 文件中。

您不应该在同一网络中的两台不同计算机上使用相同的主机名(hostname)。

在大多数系统上，主机名(hostname)映射到 `127.0.0.1` 中 `/etc/hosts`。打开文件，将旧的主机名(hostname)更改为新的主机名(hostname)。

```shell
127.0.0.1   localhost
127.0.0.1   Ubuntu-20

# The following lines are desirable for IPv6 capable hosts
::1     localhost ip6-localhost ip6-loopback
ff02::1 ip6-allnodes
ff02::2 ip6-allrouters
```

如果在云实例上运行 `Ubuntu`，并且 `cloud-init` 已安装软件包，则还需要编辑该 `/etc/cloud/cloud.cfg` 文件。通常，该软件包通常默认安装在云提供商提供的映像中，并且用于处理云实例的初始化。

如果系统上存在该文件，请打开它：

```shell
sudo vim /etc/cloud/cloud.cfg
```

搜索 `preserve_hostname`，并将值从更改 `false` 为 `true`：

```bash
# This will cause the set+update hostname module to not operate (if true)
preserve_hostname: true
```

保存文件并关闭编辑器。

## 5. 验证更改

要验证主机名(hostname)是否已完全更改，请输入以下 `hostnamectl` 命令：

```shell
hostnamectl
```

您的新主机名(hostname)将显示在终端上：

```text
   Static hostname: Ubuntu-20
         Icon name: computer-vm
           Chassis: vm
        Machine ID: XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
           Boot ID: YYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYY
    Virtualization: kvm
  Operating System: Ubuntu 20.04.4 LTS
            Kernel: Linux 5.4.0-47-generic
      Architecture: x86-64
```

## 6. 结论

我已向您展示了如何在 `Ubuntu 20.04` 安装上轻松更改主机名(hostname)而不重启机器。

有多种原因可能导致您需要更改主机名(hostname)。最常见的是在创建实例后自动设置主机名(hostname)的情况。

## 7. 参考文章

1. `如何在Ubuntu 20.04上更改主机名(hostname)`

    [https://www.iplayio.cn/post/309740](https://www.iplayio.cn/post/309740)
