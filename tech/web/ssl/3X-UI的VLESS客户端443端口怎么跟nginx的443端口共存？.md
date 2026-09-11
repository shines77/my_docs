
## 问题

3X-UI 建立的 VLESS 客户端的 443 端口怎么跟 nginx 的 443 端口共存？

## 回答

（来自 DeepSeek v4.0 或 v4.1）

让 3x-ui 的 VLESS 客户端与 Nginx 共用 443 端口，核心思路是 **SNI 分流**：由前端统一接收 443 端口流量，再根据 TLS 握手时的域名（SNI）决定转发给 Xray 还是 Nginx。

推荐 **Nginx Stream 方案**，它更通用，不修改 Xray 核心逻辑，稳定性好。

### 🎯 方案一：Nginx Stream SNI 分流（推荐）

Nginx 的 `stream` 模块只读取 SNI 信息，不解密流量，然后根据域名把 TCP 连接转发到不同后端。你需要 Nginx 已编译 `stream` 和 `stream_ssl_preread` 模块（Ubuntu 官方源通常默认包含）。

**配置思路：**

1. **Xray**：VLESS 入站监听本地端口（如 `127.0.0.1:10000`），TLS 由 Nginx 负责。
2. **Nginx**：监听公网 443，通过 `map` 指令将不同域名转发到 Xray 或你的网站后端。

**Nginx Stream 配置示例**

Nginx 的配置有严格的层级要求，stream 块必须放在主配置文件的顶层（与 http、events 同级），而不是放在 http 块内部，也不能直接放进 `/etc/nginx/conf.d/` 目录下（因为 conf.d/ 里的文件通常是被 http 块 include 的）。

所以 `stream` 配置不能放在 `/etc/nginx/conf.d/stream.conf` 里面，因为要放到跟 http {} 同一层的顶层里，

所以建议放到 `/etc/nginx/nginx.conf` 文件里。

```bash
nano /etc/nginx/nginx.conf
```

```nginx
stream {
    # 根据 SNI 域名分流
    map $ssl_preread_server_name $backend {
        vless.yourdomain.com   xray_backend;  # VLESS 域名 -> Xray
        www.yourdomain.com     web_backend;   # 网站域名 -> Nginx HTTP
        default                web_backend;   # 其他流量默认扔给 Nginx 或 Xray（或拒绝）
    }

    upstream xray_backend {
        server 127.0.0.1:10000;  # 对应 Xray 监听的本地端口
    }
    upstream web_backend {
        server 127.0.0.1:8443;   # 对应 Nginx HTTP 服务的本地端口
    }

    server {
        listen 443;
        listen [::]:443;
        ssl_preread on;          # 开启 SNI 预读
        proxy_pass $backend;     # 按 map 结果转发
    }
}
```

```nginx
stream {
    # 根据 SNI 域名分流
    map $ssl_preread_server_name $backend {
        vless.coding-fans.cn   xray_backend;  # VLESS 域名 -> Xray
        www.coding-fans.cn     web_backend;   # 网站域名 -> Nginx HTTP
        coding-fans.cn         web_backend;
        api.coding-fans.cn     web_backend;
        director.coding-fans.cn         web_backend;
        ai-director.coding-fans.cn      web_backend;
        ai-video-director.coding-fans.cn      web_backend;
        default                xray_backend;  # 这里其他的流量一定要默认扔给 Xray（或拒绝）
    }

    upstream xray_backend {
        server 127.0.0.1:10000;  # 对应 Xray 监听的本地端口
    }
    upstream web_backend {
        server 127.0.0.1:8443;   # 对应 Nginx HTTP 服务的本地端口
    }

    server {
        listen 443;
        listen [::]:443;
        ssl_preread on;          # 开启 SNI 预读
        proxy_pass $backend;     # 按 map 结果转发
    }
}
```

修改你的 nginx 的域名配置：

```bash
vim /etc/nginx/sites-available/coding-fans.cn
```

把 SSL 443 改为 127.0.0.1:8443

```nginx

    # 绑定到所有 IP 的 443 端口
    listen 443 ssl http2;
    # listen [::]:443 ssl http2;
    server_name coding-fans.cn www.coding-fans.cn vless.coding-fans.cn;

改为：

    # 绑定到所有 IP 的 443 端口
    listen 127.0.0.1:8443 ssl http2;
    listen [::1]:8443 ssl http2;
    server_name coding-fans.cn www.coding-fans.cn vless.coding-fans.cn;
```

### 🎯 方案二：Xray 回落给 Nginx

让 Xray 监听公网 443 并处理 TLS，非 VLESS 流量通过 `fallbacks` 功能“回落”给本地 Nginx 端口。

（好像这个办法行不通，至少目前这个设置，可能由于版本不同，暂时用不了。）

**配置思路：**

1. **Xray**：监听 443，TLS 证书配在 Xray 里。`fallbacks` 指向 Nginx 的本地端口（如 8080）。
2. **Nginx**：只监听本地端口（如 `127.0.0.1:8080`），不再占用公网 443。

**Xray 入站配置片段**：

只需要在 "settings" 字段内加入 `fallbacks` 即可：

```json
    // 非 VLESS 流量回落给 Nginx
    "fallbacks": [
      { "dest": 8080 }
    ],
    "fallbacks": [
      {
        "alpn": "",
        "dest": 8080,
        "name": "nginx",
        "path": "",
        "xver": 0
      }
    ],
```      

例如：

```json
{
  "inbounds": [{
    "port": 443,
    "protocol": "vless",
    "settings": {
      "clients": [{ "id": "你的UUID", "flow": "xtls-rprx-vision" }],
      // 非 VLESS 流量回落给 Nginx
      "fallbacks": [
        { "dest": 8080 }
      ]
    },
    "streamSettings": {
      "network": "tcp",
      "security": "tls",
      "tlsSettings": {
        "certificates": [{ "certificateFile": "/path/cert.pem", "keyFile": "/path/key.pem" }]
      }
    }
  }]
}
```

**Nginx 配置**：只需 `listen 127.0.0.1:8080;`（注意不要监听 443），正常提供网站服务即可。

```bash
vim /etc/nginx/sites-available/coding-fans.cn
```

### 💡 选择建议

| 对比项 | 方案一（Nginx Stream） | 方案二（Xray 回落） |
| :--- | :--- | :--- |
| **TLS 处理方** | Nginx | Xray |
| **灵活性** | 高，Nginx 可轻松代理多个网站 | 中，流量默认回落到单一 Nginx 端口 |
| **证书管理** | Nginx 统一管理 | Xray 管理 |
| **适用场景** | 已有 Nginx 且网站较多 | 偏好 Xray 原生回落逻辑 |

如果服务器上已经跑着 Nginx 网站，**方案一** 是最平滑的选择。配置完成后记得在 3x-ui 面板中将 VLESS 入站的监听端口改为本地端口（如 `10000`），并在客户端连接信息中保持域名和端口（443）不变。

--------------------------------------------------

## 问题

nginx 里 怎么设置监听 127.0.0.1:8443

## 回答

在 Nginx 里监听 `127.0.0.1:8443`，通常有两种场景：**Stream 模块的后端** 和 **普通 HTTP 服务的 server 块**。下面分别说明。

## 一、作为普通 HTTP/HTTPS 服务监听

这是最常见的用法，让 Nginx 网站只接受本机请求（比如给 Xray 回落或 Stream 转发用）。

**编辑配置文件**，比如 `/etc/nginx/sites-available/yoursite`：

```nginx
server {
    listen 127.0.0.1:8443;
    listen [::1]:8443;          # 如需 IPv6 本机监听

    server_name _;              # 因为是本机内部访问，域名可随意

    root /var/www/html;
    index index.html;

    location / {
        try_files $uri $uri/ =404;
    }
}
```

如果这里要跑 **HTTPS**（比如后端也是 TLS），则需要加证书：

```nginx
server {
    listen 127.0.0.1:8443 ssl;
    listen [::1]:8443 ssl;

    server_name yourdomain.com;

    ssl_certificate     /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    root /var/www/html;
    index index.html;
}
```

> ⚠️ 注意：如果外层已经由 Nginx Stream 做了 SNI 分流，Stream 转发过来的是**原始 TLS 流量**，这里的 `server` 块**必须加 `ssl`**，否则会握手失败。

## 二、作为 Stream 模块的 upstream 后端

如果在方案一的 `stream {}` 块里，`web_backend` 指向 `127.0.0.1:8443`，那么对应承接流量的 Nginx HTTP 服务就要监听这个地址。写法同上第一条。

Stream 侧的 upstream 写法：

```nginx
stream {
    upstream web_backend {
        server 127.0.0.1:8443;
    }
    ...
}
```

## 三、完整配置位置说明

Ubuntu 上 Nginx 配置结构：

| 路径 | 用途 |
|------|------|
| `/etc/nginx/nginx.conf` | 主配置，`stream {}` 块可放这里 |
| `/etc/nginx/sites-available/` | 各站点配置 |
| `/etc/nginx/sites-enabled/` | 启用站点的软链接 |
| `/etc/nginx/conf.d/` | 额外配置片段，常放 stream 配置 |

**推荐做法**：把 stream 配置单独放 `/etc/nginx/conf.d/stream.conf`，把 HTTP 站点放 `sites-available`。

## 四、启用并测试

```bash
# 测试语法
sudo nginx -t

# 重载配置
sudo systemctl reload nginx
```

如果报 `unknown directive "stream"`，说明 Nginx 没编译 stream 模块，需要：

```bash
# 检查是否支持
nginx -V 2>&1 | grep -o with-stream

# Ubuntu 官方源一般已内置，若没有可安装完整版
sudo apt install nginx-full
```

## 五、常见问题

1. **端口被占用**：`sudo ss -tlnp | grep 8443` 查看谁在用。
2. **只监听本机却无法访问**：`127.0.0.1` 只能本机访问，外部访问必然失败，这是正常的。要外部访问就改成 `listen 8443;`。
3. **SELinux/防火墙**：只监听回环地址不经过防火墙，一般无需额外放行。
4. **证书路径权限**：Nginx 以 `www-data` 运行，证书文件需可读。

如果你是在做 3x-ui + Nginx Stream 分流，通常需要 **两个 server 块**：一个 `listen 127.0.0.1:8443 ssl`（承接网站 HTTPS 流量），一个 stream 的 `listen 443`（对外分流）。把这两个配好，`nginx -t` 通过后 reload 即可。

---------------------------------------------------

## 问题

"stream" directive is not allowed here in /etc/nginx/conf.d/stream.conf:2

## 回答

这个报错是因为 **`stream` 块被放在了它不被允许的位置**。Nginx 的配置有严格的层级要求，`stream` 块必须放在**主配置文件的顶层（与 `http`、`events` 同级）**，而不是放在 `http` 块内部，也不能直接放进 `/etc/nginx/conf.d/` 目录下（因为 `conf.d/` 里的文件通常是被 `http` 块 include 的）。

你需要分两步调整：

### 🔍 第一步：确认 Stream 模块已加载

先确认你的 Nginx 是否支持 Stream 模块。执行：

```bash
nginx -V 2>&1 | grep -o with-stream
```

*   如果没有输出，说明模块未编译进去，需要安装 `nginx-extras` 或重新编译。
*   如果有输出（比如 `with-stream=dynamic`），说明模块是动态加载的，你需要在 `nginx.conf` 顶部的 `events` 之前手动加上加载模块的指令。

```nginx
load_module /usr/lib/nginx/modules/ngx_stream_module.so;
```

（路径可能因系统而异，可以用 `dpkg -L nginx` 或 `find` 查找 `ngx_stream_module.so` 的位置。）

### 📂 第二步：把 `stream` 块放到正确的位置

`/etc/nginx/conf.d/` 里的配置文件是被主配置文件的 `http {}` 块包含的。所以你把 `stream {}` 写在这里，就等于写在了 `http` 块内部，Nginx 就会报这个错。

**正确的做法是：**

1.  **把 `stream` 块移出 `conf.d/`**：直接编辑 `/etc/nginx/nginx.conf`，在 `http {}` 块**外面**，与 `events {}` 和 `http {}` **同级**的位置，写入你的 `stream` 配置。
2.  **或者，在主配置里 include 一个独立的 stream 配置**：在 `nginx.conf` 顶层的 `events` 和 `http` 之间，添加一行 `include /etc/nginx/stream.conf;`（文件名随意），然后把你的 `stream {}` 块写在这个新文件里。

配置结构示意：

```nginx
# 全局块
load_module modules/ngx_stream_module.so; # 如果需要动态加载

events {
    worker_connections 1024;
}

stream {                     # <-- 必须在这里！
    # 你的 SNI 分流配置
    map $ssl_preread_server_name $backend {
        vless.yourdomain.com xray_backend;
        default web_backend;
    }
    # ... 其他 stream 配置
}

http {
    # 你的网站配置
    include /etc/nginx/conf.d/*.conf; # 这里 include 进来的文件不能包含 stream 块
}
```

### 💡 关于你之前看到的 `include` 写法

有些教程里会写 `include /etc/nginx/conf.d/*.conf;` 在 `stream` 块里面，那是为了把单独的 stream 配置文件（比如叫 `xxx.streamconf`）汇聚起来。但你现在的错误在于 **`stream` 块本身的位置不对**，而不是 include 的路径不对。所以关键是先把 `stream` 块从 `conf.d/` 目录里拿出来，放到 `nginx.conf` 的顶层。

