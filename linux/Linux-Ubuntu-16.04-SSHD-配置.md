# Ubuntu SSHD 配置

`Ubuntu 16.04` 或更高。

配置文件：

```bash
vim /etc/ssh/sshd_config
```

启用下来配置：

```bash
Port 22
PermitRootLogin yes
```

```bash
# 重启 sshd 服务
systemctl restart sshd

# 查看 sshd 状态
systemctl status sshd
```

旧版 `Ubuntu` ：

```bash
# 重启 sshd 服务
service sshd restart

# 查看 sshd 状态
service sshd status
```
