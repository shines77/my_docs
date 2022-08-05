# Ubuntu 使用 ClamAV 杀毒软件

## 0. 简介

`ClamAV` 是一个在命令行下查毒软件，它不将杀毒作为主要功能，默认只能查出您计算机内的病毒，但是无法清除，只能删除感染的文件。

## 1. 安装

```shell
$ sudo apt-get install clamav clamav-daemon
```

查看版本：

```shell
$ clamscan --version

ClamAV 0.103.6/26618/Fri Aug  5 15:53:26 2022
```

## 2. 升级病毒库

```shell
$ freshclam

Sat Aug  6 02:32:00 2022 -> ClamAV update process started at Sat Aug  6 02:32:00 2022
Sat Aug  6 02:32:00 2022 -> ^Your ClamAV installation is OUTDATED!
Sat Aug  6 02:32:00 2022 -> ^Local version: 0.103.6 Recommended version: 0.103.7
Sat Aug  6 02:32:00 2022 -> DON'T PANIC! Read https://docs.clamav.net/manual/Installing.html
Sat Aug  6 02:32:00 2022 -> daily.cvd database is up-to-date (version: 26618, sigs: 1994493, f-level: 90, builder: raynman)
Sat Aug  6 02:32:00 2022 -> main.cvd database is up-to-date (version: 62, sigs: 6647427, f-level: 90, builder: sigmgr)
Sat Aug  6 02:32:00 2022 -> bytecode.cvd database is up-to-date (version: 333, sigs: 92, f-level: 63, builder: awillia2)
```

## 3. 配置文件

`clamd` 的系统设置文件位于 `/etc/clamav/clamd.conf`，设置文件的参数设置方式与说明可参考 `clamd.conf` 的在线说明。

```bash
# 查阅 clamd.conf 的在线说明文件
$ man clamd.conf
```

配置文件：

```bash
$ vim /etc/clamav/clamd.conf

# 配置内容
User clamav
MaxThreads 2
```

## 4. 扫描病毒

在使用 `clamdscan` 进行扫毒之前，可以先测试与 `clamd` 的连接是否正常：

```bash
# 检查与 clamd 的连接是否正常
$ clamdscan -p 3

# 如果连接成功则显示
PONG
```

从根目录 `/` 开始，扫描所有目录：

```bash
# 只扫描感染的文件，不删除文件
$ sudo clamscan --infected --recursive /

# 扫描感染的病毒，并删除感染的文件，慎用。
$ sudo clamscan --infected --remove --recursive /
```

`--remove` 自动删除感染病毒的文件，慎用。

## 5. clamav-daemon 系统服务

启动，查询等：

```bash
# 重启 clamav-daemon 系统服务
$ systemctl restart clamav-daemon

# 查看 clamav-daemon 服务状态
$ systemctl status clamav-daemon
```

## 6. 查看 clamd 状态

```bash
# 查看 clamd daemon 状态
clamdtop
```

## 7. 卸载 clamav

```shell
$ sudo apt-get remove clamav clamav-daemon
```
