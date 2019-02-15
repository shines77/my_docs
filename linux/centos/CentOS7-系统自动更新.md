# CentOS 7 每天自动更新

## 1. 准备

首先，我们要先手动更新所有预先安装的软件：

```shell
yum -y update
```

## 2. 自动更新

`CentOS 7` 中可以使用下列工具进行程序的自动执行：

```
cron、anacron、at 和 batch
```

其中 `cron` 和 `anacron` 用来定期重复执行指令，`At` 和 `batch` 则用来在特定时间执行一次性的指令。

我们将会使用 `cron` 和 `anacron`，两者的分别这里不细表了，现在我们现在先安装 `cron` （`cronie`）：

```shell
yum -y install cronie yum-cron
```

其中 `yum-cron` 的作用是自动下载更新，配置文件是 `/etc/yum/yum-cron.conf`，但在默认设置里，并不会安装更新文件，需要修改 `apply_updates = no` 为 `yes`，才会安装更新文件。

而 `cronie` 每天会执行一次 `/etc/cron.daily/0yum-daily.cron`，这个脚本会执行一次上面 `yum-cron` 的配置文件 `/etc/yum/yum-cron.conf`，来下载更新文件（或安装安装更新）。

所以，如果我们想实现系统的自动更新，需要修改如下配置：

```shell
vim /etc/yum/yum-cron.conf
```

并确认配置文件中的下列值：

```bash
update_messages = yes
download_updates = yes
apply_updates = yes
```

## 3. 启动服务

最后，启动 `crond` 和 `yum-cron` 服务：

```shell
systemctl start crond
systemctl start yum-cron
```

## 4. 参考文章

1. [`CentOS 7 每天自动更新`](https://www.cnblogs.com/k98091518/p/6955234.html)

    [https://www.cnblogs.com/k98091518/p/6955234.html](https://www.cnblogs.com/k98091518/p/6955234.html)
