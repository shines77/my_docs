# Let's Encrypt 免费 SSL 证书配置指南

## 前言

2026 年了，网站还不上 https ？

- 浏览器会标记为 "不安全"
- SEO 排名受影响
- 无法使用 HTTP/2、HTTP/3
- 用户数据传输有风险

以前 SSL 证书 很贵，现在有了 Let’s Encrypt，完全免费，自动续期，没有理由不用。

## 1. Let's Encrypt 简介

### 1.1 什么是 Let's Encrypt

Let's Encrypt 是一个免费、自动化、开放的证书颁发机构（CA），由非营利组织 ISRG 运营。

特点：

- 完全免费
- 自动化颁发和续期
- 被所有主流浏览器信任
- 单域名/泛域名都支持

### 1.2 证书类型

| 类型 | 说明 | 验证方式 |
| ---- | --- | ------- |
| 单域名 | 只对一个域名有效 | HTTP/DNS |
| 多域名(SAN) | 多个域名共用一个证书 | HTTP/DNS |
| 泛域名 | *.example.com | 仅 DNS |

### 1.3 验证方式

HTTP-01 验证：

- 在网站目录放置特定文件
- Let's Encrypt 访问验证
- 需要 80 端口可访问

DNS-01 验证：

- 添加特定的 DNS TXT 记录
- Let's Encrypt查询验证
- 适合泛域名、无法开放 80 端口的场景

## 2. Certbot 安装与使用

### 2.1 安装 Certbot

```bash
# Ubuntu/Debian
apt update
apt install certbot

# CentOS/RHEL
yum install epel-release
yum install certbot

# 或使用 snap（推荐，版本更新）
snap install --classic certbot
ln -s /snap/bin/certbot /usr/bin/certbot
```

### 2.2 获取证书 (HTTP 验证)

**Webroot 模式（已有 Web 服务器）：**

```bash
# 指定网站根目录, 例如: /var/www/html
certbot certonly --webroot -w /var/www/html -d example.com -d www.example.com
```

然后要输入一个 email 。

**独立运行模式（没有 Web 服务器时）：**

会启动 Web 服务，并监听 80 端口。

```bash
# 需要 80 端口空闲
certbot certonly --standalone -d example.com -d www.example.com
```

需要输入一个 email，跟前面类似。

**Nginx 插件模式（自动配置）：**

```bash
# 安装插件
apt install python3-certbot-nginx

# 自动获取证书并配置 Nginx
certbot --nginx -d example.com -d www.example.com
```

### 2.3 获取证书（DNS验证）

```bash
# 泛域名证书
certbot certonly --manual --preferred-challenges dns -d "*.example.com" -d example.com
```

执行后会提示添加 DNS TXT 记录：

```bash
Please deploy a DNS TXT record under the name:
_acme-challenge.example.com
with the following value:
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

添加记录后等待 DNS 生效，然后继续。

### 2.4 证书文件说明

证书存放在 `/etc/letsencrypt/live/example.com/`：

| 文件 | 说明 | 用途 |
| ---- | :--: | :--: |
| cert.pem | 域名证书 | - |
| chain.pem | 中间证书链 | - |
| fullchain.pem | 完整证书链 | Nginx ssl_certificate |
| privkey.pem | 私钥 | Nginx ssl_certificate_key |

## 3. Nginx https 配置

### 3.1 基础配置

```nginx
server {
    listen 80;
    listen [::]:80
    server_name example.com www.example.com;
    
    # HTTP 重定向到 HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    listen [::]:443 ssl http2;
    server_name example.com www.example.com;
    
    # SSL 证书
    ssl_certificate /etc/letsencrypt/live/example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/example.com/privkey.pem;
    
    # SSL 配置
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;
    
    # HSTS（可选，强制 HTTPS）
    add_header Strict-Transport-Security "max-age=31536000" always;
    
    root /var/www/html;
    index index.html;

    # 访问日志
    access_log /var/log/nginx/example.com_access.log;
    error_log /var/log/nginx/example.com_error.log;
    
    location / {
        try_files $uri $uri/ =404;
    }

    # 错误页面
    error_page 404 /404.html;
    location = /404.html {
        root /var/www/html;
    }
}
```

### 3.2 安全加固配置

```nginx
# /etc/nginx/conf.d/ssl.conf

# SSL 会话缓存
ssl_session_cache shared:SSL:10m;
ssl_session_timeout 1d;
ssl_session_tickets off;

# 现代加密套件
# ssl_protocols TLSv1.2 TLSv1.3;
# ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384;
# ssl_prefer_server_ciphers off;

# OCSP Stapling
ssl_stapling on;
ssl_stapling_verify on;
ssl_trusted_certificate /etc/letsencrypt/live/example.com/chain.pem;

# DNS
resolver 8.8.8.8 8.8.4.4 valid=300s;
resolver_timeout 5s;

# DH 参数（可选，增强安全性）
# openssl dhparam -out /etc/nginx/dhparam.pem 2048
# ssl_dhparam /etc/nginx/dhparam.pem;
```

#### 3.2.1 加分项：配置 HSTS

之前提到过得到 A 的方法，那么 A+ 呢?事实证明，SSL Labs 就会给一个 A+，当你有了一个叫做 HSTS (Hypertext Strict Transport Security， 超文本严格传输安全) 的特性。

#### 3.2.2 什么是 HSTS？

本质上，这是一个 HTTP 头，你可以添加到你的请求，告诉浏览器总是通过 HTTPS 访问这个站点。即使他们最初是通过 HTTP 访问的，也总是重定向到 HTTPS。

然而，这实际上有一点危险，因为如果你的 SSL 配置中断或证书过期，那么访问者将无法访问该站点的纯 HTTP 版本。你还可以做一些更高级的事情。就是将你的站点添加到预加载列表中。Chrome 和火狐浏览器都有一个列表，所以如果你注册了，他们就永远不会通过 HTTP 访问你的网站。

### 3.3 测试配置

```bash
# 测试 Nginx 配置
nginx -t

# 重载配置
nginx -s reload

# 测试 SSL
curl -I https://example.com
```

## 4. 自动续期

### 4.1 证书有效期

Let's Encrypt 证书有效期是 90 天，建议提前 30 天续期。

### 4.2 手动续期

```bash
# 续期所有证书
certbot renew

# 测试续期（不真正执行）
certbot renew --dry-run
```

### 4.3 自动续期

方法 1：Cron 定时任务

```bash
# crontab -e
0 3 * * * certbot renew --quiet --post-hook "nginx -s reload"
```

方法 2：Systemd Timer（推荐）

Certbot 安装后通常自带 timer：

```bash
# 查看 timer 状态
systemctl status certbot.timer

# 启用 timer
systemctl enable certbot.timer
systemctl start certbot.timer

# 查看下次执行时间
systemctl list-timers | grep certbot
```

### 4.4 续期钩子

```bash
# 续期成功后执行的命令
certbot renew --post-hook "systemctl reload nginx"

# 或在配置文件中设置
# /etc/letsencrypt/renewal/example.com.conf
[renewalparams]
post_hook = systemctl reload nginx
```

## 参考资料

- [Let’s Encrypt官方文档](https://letsencrypt.org/docs/)
- [Certbot文档](https://certbot.eff.org/docs/)
- [Mozilla SSL Configuration Generator](https://ssl-config.mozilla.org/)
- [Let's Encrypt 免费证书与 HTTPS 配置完全指南](https://blog.csdn.net/zhangxianhau/article/details/155909098)
- [HTTPS 基础原理和配置--(3)](https://blog.csdn.net/east4ming/article/details/129037354)
