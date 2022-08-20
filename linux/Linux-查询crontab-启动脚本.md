# Linux 查询 crontab 启动脚本

一般病毒木马执行喜欢用 `crontab`，显示所有用户的 `crontab` 的内容:

```bash
for u in `cat /etc/passwd | cut -d":" -f1`;do crontab -l -u $u;done
```

显示结果：

```text
no crontab for root
no crontab for daemon
no crontab for bin
no crontab for sys
no crontab for sync
no crontab for games
no crontab for man
no crontab for lp
no crontab for mail
no crontab for news
no crontab for uucp
no crontab for proxy
no crontab for www-data
no crontab for backup
no crontab for list
no crontab for irc
no crontab for gnats
no crontab for nobody
no crontab for systemd-network
no crontab for systemd-resolve
no crontab for systemd-timesync
no crontab for messagebus
no crontab for syslog
no crontab for _apt
no crontab for uuidd
no crontab for tcpdump
no crontab for ntp
no crontab for sshd
no crontab for systemd-coredump
no crontab for _chrony
no crontab for admin
no crontab for ocserv
no crontab for clamav
```
