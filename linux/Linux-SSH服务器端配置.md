# Linux SSH 服务器端配置

## 摘要

SSH 是一种用于加密登录和安全网络服务的网络协议，由 Tatu Ylönen 于 1995 年设计以替代不安全的协议。它通过公钥加密确保数据传输安全，主要用于远程登录、命令执行和文件传输。SSH 协议包括传输层、用户认证和连接协议三部分，支持 SSH1 和 SSH2 版本。安装和配置 SSH 服务器需修改端口、禁用密码认证和加固私钥权限。通过密钥认证、防火墙规则和定期日志检查可提升安全性。自动化运维中，SSH 用于远程命令执行、文件传输和服务器管理，提高效率和可靠性。

## 1. SSH 概述

SSH（Secure Shell）是一种网络协议，用于计算机之间的加密登录和其他安全网络服务。它提供了在网络中传输数据的加密方式，确保了数据传输的安全性。

### 1.1 SSH 的历史与发展

SSH 协议最初是由芬兰赫尔辛基大学的 Tatu Ylönen 于 1995 年设计的，目的是为了替代早期的网络协议如 Telnet 和 rlogin，这些协议在传输数据时不提供加密，因此存在安全风险。

### 1.2 SSH 的工作原理

SSH 协议通过使用公钥和私钥对会话进行加密。客户端向服务器发送公钥，服务器使用公钥加密会话密钥，然后将其发送回客户端。客户端和服务器使用会话密钥来加密所有后续的通信。

```bash
# 生成 SSH 密钥对
ssh-keygen -t rsa -b 4096
```

### 1.3 SSH 的应用场景

SSH 常用于远程登录服务器，执行命令，以及传输文件。它也可以用于隧道技术，允许用户安全地通过不安全的网络进行数据传输。

```bash
# 使用 SSH 登录远程服务器
ssh user@remote_host

# 使用 SSH 传输文件
scp /path/to/local/file user@remote_host:/path/to/remote/file
```

## 2. SSH协议基础

SSH 协议是建立在应用层和传输层之上的协议，它主要由三个主要部分组成：传输层协议、用户认证协议和连接协议。

### 2.1 传输层协议

传输层协议为 SSH 会话提供服务器认证、数据加密、数据完整性验证和压缩。它使用公钥加密算法来建立安全连接。

```bash
# 示例：查看 SSH 传输层使用的加密算法
ssh -v
```

### 2.2 用户认证协议

用户认证协议用于验证用户的身份。SSH 支持多种认证方法，包括密码认证、公钥认证和基于主机的认证。

```bash
# 示例：使用 SSH 密钥进行用户认证
ssh -i /path/to/private/key user@remote_host
```

### 2.3 连接协议

连接协议用于多个通道之间的交互，这些通道用于传输终端会话、文件传输和转发 X11 应用程序等。

```bash
# 示例：通过 SSH 创建一个用于文件传输的通道
ssh -L 9999:localhost:9999 user@remote_host
```

### 2.4 SSH 协议版本

SSH 协议有两个主要版本：SSH1 和 SSH2。SSH2 是 SSH1 的改进版本，提供了更强的加密算法和更好的安全性。

```bash
# 示例：指定使用 SSH2 协议版本
ssh -o Protocol=2 user@remote_host
```

## 3. SSH 服务器安装与配置

为了能够远程安全地管理 Linux 服务器，通常需要安装和配置 SSH 服务器。以下是在不同 Linux 发行版上安装和配置 SSH 服务器的步骤。

### 3.1 安装 SSH 服务器

在 Debian/Ubuntu 上：

```bash
sudo apt-get update
sudo apt-get install openssh-server
```

在 CentOS/RHEL 上：

```bash
sudo yum install openssh-server
```

### 3.2 配置 SSH 服务器

SSH 服务器的配置文件通常位于 `/etc/ssh/sshd_config`。以下是一些常见的配置选项：

- **Port**：指定 SSH 服务器监听的端口号。
- **PermitRootLogin**：是否允许 root 用户登录。
- **PasswordAuthentication**：是否允许使用密码认证。
- **RSAAuthentication**：是否允许使用 RSA 密钥认证。

```bash
# 示例：编辑 SSH 服务器配置文件
sudo vim /etc/ssh/sshd_config

# 修改以下配置项（根据需要）
Port 22
PermitRootLogin yes
PasswordAuthentication yes
RSAAuthentication yes

X11Forwarding yes

# 实际的设置一般是这样的, 允许PAM,
# 但关闭 PasswordAuthentication 和 KbdInteractiveAuthentication
PasswordAuthentication no
KbdInteractiveAuthentication no
UsePAM yes

# 重启SSH服务使配置生效
sudo systemctl restart ssh
```

关于 `UsePAM` 的解释：

```bash
# Set this to 'yes' to enable PAM authentication, account processing,
# and session processing. If this is enabled, PAM authentication will
# be allowed through the KbdInteractiveAuthentication and
# PasswordAuthentication.  Depending on your PAM configuration,
# PAM authentication via KbdInteractiveAuthentication may bypass
# the setting of "PermitRootLogin prohibit-password".
# If you just want the PAM account and session checks to run without
# PAM authentication, then enable this but set PasswordAuthentication
# and KbdInteractiveAuthentication to 'no'.

# 中文翻译

# 将此设置为 “yes” 以启用 PAM 身份验证、账户处理，
# 以及会话处理。如果启用此功能，PAM 身份验证将
# 被允许通过 KbdInteractiveAuthentication 和
# PasswordAuthentication (密码认证)。根据您的 PAM 配置，
# 通过 KbdInteractiveAuthentication 进行的 PAM 认证可能会被绕过
# "PermitRootLogin prohibit-password" 的设置。
# 如果您只想运行 PAM 账户和会话检查，而无需
# PAM 认证，然后启用此功能，但需设置将 PasswordAuthentication
# 和 KbdInteractiveAuthentication 设置为 'no'。
```

PAM 的配置文件主要放在 `/etc/pam.d/` 目录下，每个需要认证的服务都有一个对应的文件。比如 SSH 服务的配置就在 `/etc/pam.d/sshd`。

```bash
vim /etc/pam.d/sshd
```

### 3.3 管理 SSH 服务

可以使用以下命令来管理 SSH 服务的状态：

```bash
# 启动SSH服务
sudo systemctl start ssh

# 停止SSH服务
sudo systemctl stop ssh

# 重启SSH服务
sudo systemctl restart ssh

# 检查SSH服务状态
sudo systemctl status ssh
```

### 3.4 安全加固 SSH 服务器

为了提高 SSH 服务器的安全性，可以采取以下措施：

- 修改默认端口
- 只使用 SSH 协议版本2
- 禁用 root 登录
- 禁用密码认证，使用密钥认证
- 限制用户和用户组
- 使用防火墙限制 SSH 访问

```bash
# 示例：编辑 SSH 服务器配置文件
sudo vim /etc/ssh/sshd_config

# 修改以下配置项（根据需要）

# 修改默认端口（推荐）
Port 30821
# 只使用 SSH 协议版本2（推荐）
Protocol 2
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

# 同时允许 5 个尚未登录的 SSH 联机
MaxStartups 5
# 最大登录尝试次数（默认值为6）
MaxAuthTries 5
# 最大会话数
MaxSessions 10

# 限制用户和用户组（推荐）
# 注：请修改为你的用户名和用户组

# 允许通过远程访问的用户，多个用户以空格分隔
AllowUsers user1 user2
# 允许通过远程访问的组，多个组以空格分隔
AllowGroups group1 group2

# 禁止通过远程访问的用户，多个用户以空格分隔
DenyUsers user1 user2
# 禁止通过远程访问的组，多个组以空格分隔
DenyGroups group1 group2

# 重启SSH服务使配置生效
sudo systemctl restart ssh
```

建议还是允许 root 登陆（为了方便），并且开启密码认证（也是为了方便）。当然，如果为了安全，可以两者都关闭，使用密钥登录。

#### 限制或允许远程连接的 IP

如果需要限制来路 IP 的话，可以修改 `/etc/hosts.deny` 和 `/etc/hosts.allow` 两个文件，通过添加`sshd:<IP地址或IP段>` 来限制或允许 SSH 远程连接 IP。

#### 使用防火墙限制 SSH 访问

```bash
# 示例：使用防火墙限制 SSH 访问（只允许 TCP 22 端口）
sudo iptables -A INPUT -p tcp --dport 22 -j ACCEPT
sudo iptables -A INPUT -j DROP

# 示例：使用 iptables 限制 SSH 访问（不推荐）
sudo iptables -A INPUT -p tcp --dport 30821 -s 192.168.1.0/24 -j ACCEPT
sudo iptables -A INPUT -p tcp --dport 30821 -j DROP
```

#### 设置 SSH 密钥的权限

更改 `.ssh` 目录的权限为 700，`authorized_keys` 文件的权限为 600。

确保 SSH 私钥文件的权限设置正确，防止未授权访问。

```bash
# 设置 .ssh 目录和 authorized_keys
chmod 700 ~/.ssh
chmod 600 ~/.ssh/authorized_keys

# 设置私钥文件权限
chmod 600 ~/.ssh/id_rsa
```

#### 定期更新和检查日志

定期检查 SSH 日志文件，如 `/var/log/auth.log`，以监控可疑活动。

```bash
# 查看 SSH 日志
sudo tail -f /var/log/auth.log
```

通过实施这些安全策略和优化措施，可以显著提高 SSH 服务器的安全性。

## 4. 参考文章

- [SSH客户端与服务器配置实战解析](https://my.oschina.net/emacs_9519843/blog/18723465)
- [隐秘通道：通过PAM模块实现SSH认证绕过与密码捕获](https://blog.csdn.net/weixin_29274969/article/details/158404519)
