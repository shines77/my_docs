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

## 4. 安装 3x-ui 面板

最新版是 `3.7.0`（2026/09/04），并集成 `Xray-core v26.7.28`，使用 `Go 1.27` 编译。

```bash
bash <(curl -Ls https://raw.githubusercontent.com/mhsanaei/3x-ui/master/install.sh)
```

### 4.1 选 SQLite

```bash
═══════════════════════════════════════════
     Database Selection                    
═══════════════════════════════════════════
  1) SQLite     (default — recommended for < 500 clients)
  2) PostgreSQL (recommended for high client counts / many nodes)

Choose [1]: 1  
```
### 4.2 选择面板端口

随意，随机选 `n`，自定义选 `y`，例如：25033。

```bash
Would you like to customize the Panel Port settings? (If not, a random port will be applied) [y/n]: n
```

### 4.3 SSL 证书

有域名选 1，只有 IP 选 2，自己有域名有证书选 3，选 4 则跳过 SSL 证书，默认选 2。

```bash
═══════════════════════════════════════════
     SSL Certificate Setup (RECOMMENDED)   
═══════════════════════════════════════════
SSL is strongly recommended. Skip only if a reverse proxy
or SSH tunnel handles TLS for you.
Let's Encrypt now supports both domains and IP addresses!

Choose SSL certificate setup method:
1. Let's Encrypt for Domain (90-day validity, auto-renews)
2. Let's Encrypt for IP Address (6-day validity, auto-renews)
3. Custom SSL Certificate (Path to existing files)
4. Skip SSL (advanced — behind reverse proxy / SSH tunnel only)

Note: Options 1 & 2 require port 80 open. Option 3 requires manual paths.
Note: Option 4 serves the panel over plain HTTP — only safe behind nginx/Caddy or an SSH tunnel.

Choose an option (default 2 for IP): 2
```

### 4.4 监听端口

询问是否使用 80 端口，申请证书。`acme.sh` 用 HTTP-01 方式验证域名所有权，默认使用 80 端口，这个建议不要改。

### 4.5 安装完毕

安装完成的面板信息，示例如下：

```bash
═══════════════════════════════════════════
     Panel Installation Complete!         
═══════════════════════════════════════════
Username:    Abcdefghij
Password:    0123456789
Port:        12345
WebBasePath: a1b2c3d4e5f6g7h8i9
Database:    SQLite (/etc/x-ui/x-ui.db)
Access URL:  https://114.114.114.114:12345/a1b2c3d4e5f6g7h8i9
API Token:   kjdfuidshfkjdkjfhdsiufhijfbndeoikfdjodsfhiudhfif
═══════════════════════════════════════════
```

最后是一些基本命令：

```bash
x-ui v3.7.0 installation finished, it is running now...

┌───────────────────────────────────────────────────────┐
│  x-ui control menu usages (subcommands):              │
│                                                       │
│  x-ui              - Admin Management Script          │
│  x-ui start        - Start                            │
│  x-ui stop         - Stop                             │
│  x-ui restart      - Restart                          │
│  x-ui status       - Current Status                   │
│  x-ui settings     - Current Settings                 │
│  x-ui enable       - Enable Autostart on OS Startup   │
│  x-ui disable      - Disable Autostart on OS Startup  │
│  x-ui log          - Check logs                       │
│  x-ui banlog       - Check Fail2ban ban logs          │
│  x-ui update       - Update                           │
│  x-ui legacy       - Legacy version                   │
│  x-ui install      - Install                          │
│  x-ui uninstall    - Uninstall                        │
└───────────────────────────────────────────────────────┘
```

记得先去你的 VPS 防火墙设置放行 3x-ui 面板的端口，再到浏览器打开网址。

## 5. 防火墙设置

```bash
ufw default deny incoming
ufw default allow outgoing

ufw allow 22/tcp
ufw allow 80/tcp
ufw allow 443/tcp
ufw allow 443/udp

# 面板端口
ufw allow 32451/tcp

# 订阅端口
ufw allow 2096/tcp

# SSH 端口
ufw allow 25042/tcp
ufw allow 25042/udp

# VLESS 节点端口
ufw allow 23918/tcp

# 出站 DNS
sudo ufw allow out 53/udp

# inbounds: 127.0.0.1/62798
ufw allow 62798/tcp
ufw allow 62798/udp

# 10443
ufw allow 10443/tcp
ufw allow 10443/udp

ufw enable
ufw status
```

## 6. 配置 3x-ui

使用前面给的 URL 打开 3x-ui 面板后，输入用户名和密码，进入系统。

### 6.1 添加 VLESS + Reality 节点

左边的导航菜单里点击“入站”，然后点击“添加入站”按钮。

- 备注：这里填写节点名称，例如：美国–vless–001
- 协议：这里选 vless 协议
- 其他选项不动，我们切换到 “传输” 标签

- “传输” 标签里，传输 - 选 RAW 或者叫 TCP。其他不改

- 切换到“安全”标签，安全 - 选 “Reality”
- 目标 – 点击查找目标，点击第一个，例如是："www.amd.com:443"，或者输入"www.bing.com:443"，然后点“搜寻”，如果可用，就选它。

- 其他保持默认，点“创建”按钮。

然后切换回左边的菜单，点击“客户端” –- “添加客户端”。

- 邮箱随便填一个像正常人的名字，例如："hamjimi"。
- 点击“关联入站”的输入框，选择刚才我们创建的“入站”信息。
- 点击“创建”。

至此，我们已经完成了创建 VLESS + Reality 节点。

就在我们刚创建的客户端，第一个小按钮，是“二维码”，弹出来的二维码界面，点击“订阅信息”，把它收起来，就能看到 vless 节点二维码了。

因为我们要的是节点信息的二维码，不是“订阅信息”的二维码，看到“VLESS”、“TCP”、“REALITY” 字样在前面，就是 vless 节点二维码了。

用下列的方法扫描该二维码：

- 苹果手机使用 shadowrocket 或者 karing 导入该节点
- 安卓手机使用 v2rayng 或者 karing 导入该节点
- Windows 电脑使用 v2rayn 或者 hiddify 导入该节点
- Mac 电脑使用 karing 导入该节点

### 6.2 开启 BBR 加速

在 SSH 终端里输入命令 `x-ui`，如下：

```bash
The OS release is: ubuntu

╔────────────────────────────────────────────────╗
│  3X-UI Panel Management Script                │
│  0. Exit Script                               │
│────────────────────────────────────────────────│
│  1. Install                                   │
│  2. Update                                    │
│  3. Update to Dev Channel (latest commit)     │
│  4. Update Menu                               │
│  5. Legacy Version                            │
│  6. Uninstall                                 │
│────────────────────────────────────────────────│
│  7. Reset Username & Password                 │
│  8. Reset Web Base Path                       │
│  9. Reset Settings                            │
│  10. Change Port                              │
│  11. View Current Settings                    │
│────────────────────────────────────────────────│
│  12. Start                                    │
│  13. Stop                                     │
│  14. Restart                                  │
|  15. Restart Xray                             │
│  16. Check Status                             │
│  17. Logs Management                          │
│────────────────────────────────────────────────│
│  18. Enable Autostart                         │
│  19. Disable Autostart                        │
│────────────────────────────────────────────────│
│  20. SSL Certificate Management               │
│  21. Cloudflare SSL Certificate               │
│  22. IP Limit Management                      │
│  23. Firewall Management                      │
│  24. SSH Port Forwarding Management           │
│  25. PostgreSQL Management                    │
│────────────────────────────────────────────────│
│  26. Enable BBR                               │
│  27. Update Geo Files                         │
│  28. Speedtest by Ookla                       │
╚────────────────────────────────────────────────╝

Panel state: Running
Start automatically: Yes
xray state: Running

Please enter your selection [0-28]: 26
```

选 26，然后选 1，如下：

```bash
Please enter your selection [0-28]: 26
	1. Enable BBR
	2. Disable BBR
	0. Back to Main Menu
Choose an option: 1
```

就开启了 BBR，然后按回车返回主菜单，再选 0 则退出菜单。

### 6.3 Reality 的 dest 域名怎么选？为什么不能用国内站？

创建入站时选 VLESS + Reality 协议，dest 字段填的是「伪装目标域名:端口」。Reality 的工作原理是：你的 VPS 发出的 TLS 握手数据包在外层看起来像是访问目标网站的流量，利用目标网站的真实 TLS 证书作掩护。

关键选择标准：

- **必须是全球大站，TLS 证书长期有效且稳定。** 微软、苹果、亚马逊、Cloudflare 的根域名都是好选择。它们的证书由专业团队管理，几乎不会过期或更换 CA。
- **不能选中国大陆网站。** 国内大站的 CDN 节点分布在不同地区，TLS 指纹和海外用户行为不一致，且部分网站有严格的 CDN 调度策略——来自海外 VPS 的访问可能被路由到不同节点，导致握手指纹异常。
- **不要选小网站或个人博客。** 小站可能随时换服务器、换 CDN、换证书，一旦证书指纹变了你的入站就断了，排查起来也很麻烦。

推荐 dest 列表（截至 2026 年 5 月均稳定可用）：

|dest 域名|适用场景|注意事项|
|---|---|---|
|microsoft.com|通用推荐，全球节点多|CDN 节点覆盖广，响应快|
|bing.com|通用推荐，全球节点多|CDN 节点覆盖广，响应快|
|dl.google.com|通用推荐，全球节点多|CDN 节点覆盖广，响应快|
|apple.com|适合对延迟敏感的场景|Apple 的 CDN 在某些 VPS 机房延迟偏高|
|amazon.com|适合流量较大的入站|Amazon 的 TLS 指纹特征与其他电商站不同|
|cloudflare.com|备选，TLS 稳定|Cloudflare 对自己域名的访问控制较严|

填好 dest 后点「生成密钥对」，面板会给出 private key 和 public key 两个值。Flow 选 `xtls-rprx-vision`，这是 Xray 官方推荐的 Reality 流控方式。shortId 留空即可，除非你有特殊需求需要额外混淆。

## X. 参考文章

- [2026年8月最新稳定Debian搭建VLESS+TCPRAW+REALITY+Vision节点教程](https://skylink9119.github.io/2026/07/30/2026-august-debian-vless-tcp-raw-reality-vision/)

- [3X-UI 面板安装与首次配置完整教程（VPS 2026）](https://www.chonglangbiji.com/howto/3x-ui-install-vps/)

- [什么是 3x-ui 面板？](https://3x-ui.pro/zh/)

- [Xray VLESS + REALITY 从源码编译安装：拒绝一键脚本，部署最干净的代理服务](https://vpsdeck.com/xray-source-install-vless-reality/)

- []()


