# Ubuntu 24.04 新安装需要做的基本防护

## 1. 更新/升级

```bash
apt update
apt upgrade -y
```

这是基本操作。

## 2. 修改 root 密码

因为有的轻量级服务器为了安全，默认给你的不是 root 用户，而是 ubuntu 这样的用户。

虽然也是超级管理员组，但每次都要加 sudo 前缀很麻烦。

设置 root 用户的新密码：

```bash
sudo passwd root
```

输入两遍同样的密码。

## 3. SSH 安全加固

```bash
vim /etc/ssh/sshd_config

# 只使用 SSH 协议版本2（推荐）
Protocol 2

# 修改默认端口（推荐）
Port 25042

# 禁用 root 登录（不推荐）
PermitRootLogin no

# 禁用密码认证，使用密钥认证（不推荐）
PasswordAuthentication no

# 使用 RSA 密钥认证（已废弃）
# 自 ‌OpenSSH 7.3（2016 年发布）起‌，SSH-1 协议已被彻底移除，因此该选项‌不再生效且已被弃用‌‌
RSAAuthentication yes

# 允许公钥认证
PubkeyAuthentication yes

# 保存公钥的认证文件
# Expect .ssh/authorized_keys2 to be disregarded by default in future.
AuthorizedKeysFile .ssh/authorized_keys .ssh/authorized_keys2
```

一般为了安全，是推荐使用密钥登陆的，但是为了方便，一般还是只使用密码登陆。

其他 SSH 安全加固，可参考 [Linux SSH 服务器端配置](./Linux SSH 服务器端配置.md) 。

**新更改的 SSH 端口**

需要到你的 VPS 的防火墙设置里放行该端口。

重启 SSH 服务使配置生效：

```bash
sudo systemctl restart sshd
```

但是要注意，如果修改了 SSH 端口，则需要重启一次服务器才能生效。

## 4. fail2ban

虽然我们更改了默认的 SSH 端口，但还是避免不了有人试探我们的 SSH 登陆，为了增加试探的难度，可以使用 fail2ban 来防护。

### 4.1 安装

安装 Fail2ban：

```bash
apt install fail2ban
```

安装完成后，Fail2ban 服务会自动启动。你可以用以下命令确认其运行状态：

```bash
sudo systemctl status fail2ban
```

Fail2ban 默认已经启用了对 SSH 的保护，无需任何修改就能工作。它的默认行为是：在10分钟（600秒）内，如果同一个 IP 地址有 5次 失败的 SSH 登录尝试，就会将该 IP 封禁 10 分钟。

### 4.2 配置文件

配置建议：虽然默认配置可用，但更推荐的做法是创建自定义配置文件 `/etc/fail2ban/jail.local`。这个文件会覆盖默认配置，并且在进行软件更新时不会被覆盖，更安全。

```bash
sudo cp /etc/fail2ban/jail.conf /etc/fail2ban/jail.local
```

上面这个 `jail.conf` 就是默认设置，有了 `jail.local` 则会覆盖默认设置。

修改关键参数：

```
sudo vim /etc/fail2ban/jail.local

# 白名单 IP，这些 IP 不会被封禁。建议将你自己的 IP 地址或本地网络加入，防止误封
ignoreip = 127.0.0.1/8 ::1 192.168.1.0/24

# 封禁时长（秒）。3600 为 1 小时，86400 为 1 天
# 也可以用分钟为单位，例如：10m（10分钟）
bantime = 20m

# 检查失败次数的“时间窗口”（秒），也可以用分钟为单位，例如：10m（10分钟）
findtime = 5m

# 在 findtime 定义的时间窗口内，允许的最大失败次数
maxretry = 5
```

在文件末尾，找到 [sshd] 部分，确保 SSH 防护是开启状态，并可以指定监听的端口（如果你修改了 SSH 默认的 22 号端口）

```bash
[sshd]
enabled = true
port    = ssh  # 如果改了端口，比如 2222，就写成 port = 2222
logpath = %(sshd_log)s
backend = %(sshd_backend)s

# 例如：
enabled = true
port    = 25042
logpath = /var/log/auth.log
backend = systemd
```

重启服务使配置生效：

```bash
sudo systemctl restart fail2ban
```

### 4.3 常用管理命令

配置好后，你可以通过 `fail2ban-client` 命令来管理和查看 Fail2ban 的状态。

- **查看 Fail2ban 整体状态**：

```bash
sudo fail2ban-client status
```

这个命令会列出所有已激活的监控服务（Jails），比如 sshd。

- **查看 SSH 服务的详细状态**：

这会显示当前被该服务监控并封禁的 IP 列表。

```bash
sudo fail2ban-client status sshd
```

- **手动解封一个 IP 地址**：

如果你确定某个被封的 IP 是安全的，可以手动解封。

```bash
sudo fail2ban-client unban <IP地址>
```

- **查看 Fail2ban 自己的日志**：

可以在这里看到封禁和解封的记录。

```bash
sudo tail -f /var/log/fail2ban.log
```

### 4.4 进阶：与 UFW 防火墙协同

如果你的服务器使用了 ufw 来管理防火墙规则，建议让 Fail2ban 也通过 ufw 来添加封禁规则，这样规则更统一。

你需要在 `/etc/fail2ban/jail.local` 文件中，将 [DEFAULT] 部分的 banaction 修改为：

```text
banaction = ufw
```

修改后，再次重启 Fail2ban 服务即可。

### 4.5 故障排查

- **确认 backend 设置**：Ubuntu 24.04 默认使用 systemd 来读取日志。如果你的自定义配置覆盖了它，可能会导致 Fail2ban 无法工作。请检查 `/etc/fail2ban/jail.local` 文件中是否有 `backend = systemd` 这一行，如果没有，建议加上。

- **检查 SSH 日志路径**：如果使用了非默认的 SSH 端口，请确保 [sshd] 部分的 logpath 正确，通常是 `/var/log/auth.log` 。

### 4.6 阻拦信息

想知道更想象的阻拦信息，请查阅：[fail2ban-阻拦信息查看.md](./fail2ban-阻拦信息查看.md)。
