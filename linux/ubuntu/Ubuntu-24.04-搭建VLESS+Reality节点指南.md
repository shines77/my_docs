# Ubuntu 24.04 搭建 VLESS + Reality 节点指南

## 1. VLESS + Reality

VLESS + Reality 由于不需要域名和域名证书，防封锁能力强，而成为现在主流的上网方式。

## 2. 3X-UI

3X-UI 是一个开源的 Xray 面板，用网页界面来管理 Xray-core（一个功能很全的代理内核），支持 VLESS、VMess、Trojan、Shadowsocks 等协议。它是在早期 x-ui 项目基础上做的增强分支，目前是这一系列里维护比较活跃、功能比较完整的一个。3X-UI 本身是一个 Golang 写的轻量面板，安装只需一行命令，但要安全使用需要完成：放行端口 → 安装脚本 → 获取面板登录信息 → 配置 SSL 证书 → 创建入站 → 测试连通性。

官方 github：[https://github.com/MHSanaei/3x-ui/](https://github.com/MHSanaei/3x-ui/)

核心特点大致是：

- 多协议支持——VLESS、VMess、Trojan、Shadowsocks、Wireguard 等主流协议都能配，也支持 XTLS/Reality 这类较新的抗封锁传输方式。

- Web 管理面板——不用手写 Xray 的 JSON 配置，直接在浏览器里增删入站（inbound）、生成用户、配置传输层参数。

- 多用户与流量管理——可以按用户设置流量上限、到期时间、IP 并发限制，适合自己用或分发给多人。

- 监控与运维——面板里能看流量统计、系统负载，支持 Telegram 机器人推送告警，也内置了证书申请、订阅链接生成等功能。

- 部署简单——一条安装脚本或 Docker 就能跑起来，配合 Debian/Ubuntu 这类 VPS 很常见。

- 它把 Xray 原本偏底层、要手动改配置文件的使用方式，包装成了一个可视化后台，降低了搭建和管理代理节点的门槛。

以下从购买服务器到在代理软件中导入节点，一步一个脚印教你如何搭建属于自己的 VPN 节点！

## 3. 环境准备

除了 VPS 本身，你还需要一个已经解析到 VPS IP 的域名。SSL 证书申请需要域名验证，3X-UI 面板通过 HTTPS 访问也需要域名。用免费域名（如 freenom 上拿的 .tk）也可以，前提是 DNS 能正常解析。

### 3.1 更新系统

```bash
apt-get update
apt-get upgrade -y
```

### 3.2 安装基础工具

```bash
apt-get install -y curl wget vim socat ca-certificates lsb-release gnupg ufw
```

这些工具大多数已经安装有了，前面更新过也基本是最新的了。

### 3.3 设置时区

```bash
timedatectl set-timezone Asia/Shanghai
```

对于时区，正常情况下，前面更新的过程中也设置成上海了。

### 3.4 开启 BBR

```bash
sh -c 'echo "net.core.default_qdisc=fq" >> /etc/sysctl.conf'
sh -c 'echo "net.ipv4.tcp_congestion_control=bbr" >> /etc/sysctl.conf'
sysctl -p
lsmod | grep bbr
```

如果 `lsmod | grep bbr` 有返回值，就说明 BBR 已启用。

## X. 参考文章

- [2026年8月最新稳定Debian搭建VLESS+TCPRAW+REALITY+Vision节点教程](https://skylink9119.github.io/2026/07/30/2026-august-debian-vless-tcp-raw-reality-vision/)

- [3X-UI 面板安装与首次配置完整教程（VPS 2026）](https://www.chonglangbiji.com/howto/3x-ui-install-vps/)

- [什么是 3x-ui 面板？](https://3x-ui.pro/zh/)

- []()

- []()


