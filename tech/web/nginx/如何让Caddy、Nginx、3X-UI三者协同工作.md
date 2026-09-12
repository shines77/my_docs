# 如何让 Caddy、Nginx、3x-ui 三者协同工作

## 问题

如何让 caddy、nginx、3x-ui 一起组合起来使用，3x-ui 的 REALITY 也得走443，由谁来接管 80/443 更好，三者要如何设置？至少给我两种方案，3x-ui 不能放在最前端。建议优先考虑 nginx 方案，因为 stream 模块一般源安装就能支持。

## 回答

（DeepSeek-v4.1-falsh，会话地址：[https://chat.deepseek.com/a/chat/s/65547ea3-59dd-49ce-8b3e-14aa7ceeb87a]()。）

让 Caddy、Nginx、3x-ui 三者协同工作，同时满足“REALITY 走 443”和“3x-ui 不在最前端”的要求，核心矛盾在于：**公网 443 端口只能有一个服务监听**。因此，必须由其中一个服务站在最前端做 TCP 层 SNI 分流，根据客户端 TLS 握手中的域名，将 REALITY 流量和普通 Web 流量在 TLS 握手完成之前就分开，分别转发给本地的 Xray 和 HTTP 反代引擎。

你优先考虑 Nginx 方案是对的，因为 Nginx 的 `stream` 模块（`ssl_preread`）通常随发行版源安装即可用，比 Caddy 的 `caddy-l4` 插件少一层编译依赖。

### 方案一：Nginx Stream 做前端入口（推荐）

这是最稳妥的组合。Nginx 只负责 TCP 层 SNI 分流，Caddy 退居内部做 HTTP 层的反向代理和证书管理，3x-ui 的 REALITY 入站、面板、订阅全部只监听本地。

**端口分配**（所有 `127.0.0.1` 端口都不暴露公网）：

| 组件 | 监听地址 | 说明 |
|---|---|---|
| Nginx HTTP | `0.0.0.0:80` | 前端 80 入口，唯一对外监听。http 还是交给 nginx 管理，https 自动跳转 |
| Nginx Stream | `0.0.0.0:443` | 前端 443 入口，唯一对外监听 |
| Nginx HTTPS | `127.0.0.1:8443` | Nginx 内部 HTTPS 引擎 |
| Caddy | `127.0.0.1:6443` | Caddy 内部 HTTPS 引擎 |
| ocserv | `127.0.0.1:10443` | Csico VPN Server，只供本机转发 |
| 3x-ui REALITY | `127.0.0.1:10000` | Xray 入站，只供本机转发 |
| 3x-ui 面板 | `127.0.0.1:32451` | 3x-ui 面板管理 |
| 3x-ui 订阅 | `127.0.0.1:2096` | 3x-ui 订阅服务 |

**流量路径**：

```
用户: 443 → Nginx Stream (读 SNI)
    ├─ SNI = vless.example.com → 127.0.0.1:10000 (3x-ui REALITY)
    ├─ SNI = example.com → 127.0.0.1:8443 (Nginx)
    ├─ SNI = www.example.com → 127.0.0.1:8443 (Nginx)
    ├─ SNI = m.example.com → 127.0.0.1:8443 (Nginx)
    ├─ SNI = api.example.com → 127.0.0.1:8443 (Nginx)
    ├─ SNI = site.example.com → 127.0.0.1:6443 (Caddy)
    ├─ SNI = director.example.com → 127.0.0.1:6443 (Caddy)
    └─ 其他 SNI → 127.0.0.1:10000 (3x-ui REALITY)
                    ├─ panel.example.com → 127.0.0.1:32451 (3x-ui 面板)
                    └─ sub.example.com  → 127.0.0.1:2096 (3x-ui 订阅服务)
```

系统的 443 端口由 nginx 接管，再把 3x-ui REALITY 转到 `vless.example.com`，这个域名可以在 3x-ui 的新建 “入站” 时的“基础设置”中的“分享地址策略”选自定义，然后“自定义分享地址”里填入 "vless.example.com"，即可。

然后让 Nginx 接管大部分的域名，一个个域名设置对应的映射即可。

Caddy 也一样，有一个域名就映射一个就可以了。

最后让兜底的流量都转到 3x-ui，例如：3x-ui 面板，3x-ui 订阅等，因为在 3x-ui 内部改设置比较麻烦，默认的其他流量都转给他了。

Nginx，Caddy，3x-ui 都可以自动续签域名的 SSL 签名，只不过 Caddy 续签是自动的，比较方便，但是 3x-ui 也是自动的，只有 Nginx 要自己配置，但也很简单。

**Nginx Stream 配置**

（创建 `/etc/nginx/modules-enabled/90-stream.conf`，避免直接改 `nginx.conf`）：

```nginx
stream {
    # 根据 SNI 域名分流
    map $ssl_preread_server_name $backend {
        vless.example.com        xray_backend;  # VLESS 域名 -> Xray REALITY 点域名
        www.example.com          web_backend;   # 网站域名 -> Nginx HTTPS
        example.com              web_backend;   # 网站域名 -> Nginx HTTPS
        m.example.com            web_backend;   # Nginx HTTPS
        ai.example.com           web_backend;   # Nginx HTTPS
        api.example.com          web_backend;   # Nginx HTTPS
        site.example.com         caddy_backend; # Caddy HTTPS
        director.example.com     caddy_backend; # Caddy HTTPS
        default                  xray_backend;  # 其余流量默认全部交给 Xray（或拒绝）
    }

    upstream xray_backend {
        # 在 3x-ui 里域名绑定到 panel.example.com ，这个可以在安装时设置。
        # 也可以执行命令 `x-ui`，然后选选项 20 重新配置。
        server 127.0.0.1:10000;  # 对应 Xray REALITY 监听的本地端口
    }

    upstream caddy_backend {
        server 127.0.0.1:6443;   # 对应 Caddy HTTPS 服务的本地端口
    }

    upstream web_backend {
        server 127.0.0.1:8443;   # 对应 Nginx HTTPS 服务的本地端口
    }

    # Stream 只接管 HTTPS，80 端口仍由 Nginx 负责 HTTP 到 HTTPS 的跳转
    server {
        listen 443 reuseport;
        listen [::]:443 reuseport;
        ssl_preread on;          # 开启 SNI 预读，只读 ClientHello，不完成握手
        proxy_pass $backend;     # 按 map 结果转发
    }
}
```

`reuseport` 的作用是：**允许同一台机器上的多个  socket 绑定到同一个 IP:端口组合**，消除 accept 锁竞争，并由内核在它们之间做负载均衡，能提高 443 端口的吞吐量。

实操版：

```nginx
stream {
    # 根据 SNI 域名分流
    map $ssl_preread_server_name $backend {
        vless.coding-fans.cn        xray_backend;  # VLESS 域名 -> Xray REALITY 点域名
        www.coding-fans.cn          web_backend;   # 网站域名 -> Nginx HTTP
        coding-fans.cn              web_backend;   # Nginx HTTPS
        m.coding-fans.cn            web_backend;   # Nginx HTTPS
        ai.coding-fans.cn           web_backend;   # Nginx HTTPS
        api.coding-fans.cn          web_backend;   # Nginx HTTPS
        site.coding-fans.cn         caddy_backend; # Caddy HTTPS
        director.coding-fans.cn     caddy_backend; # Caddy HTTPS
        default                     xray_backend;  # 其余流量默认全部交给 Xray（或拒绝）
    }

    upstream xray_backend {
        # 在 3x-ui 里域名绑定到 panel.coding-fans.cn，这个可以在安装时设置。
        # 也可以执行命令 `x-ui`，然后选选项 20 重新配置。
        server 127.0.0.1:10000;  # 对应 Xray REALITY 监听的本地端口
    }

    upstream caddy_backend {
        server 127.0.0.1:6443;   # 对应 Caddy HTTPS 服务的本地端口
    }

    upstream web_backend {
        server 127.0.0.1:8443;   # 对应 Nginx HTTPS 服务的本地端口
    }


    server {
        listen 443 reuseport;
        listen [::]:443 reuseport;
        ssl_preread on;          # 开启 SNI 预读，只读 ClientHello，不完成握手
        proxy_pass $backend;     # 按 map 结果转发
    }
}
```

**Caddyfile**（Caddy 只监听本地 6443）：

```caddyfile
{
    https_port 6443
}

site.example.com {
    reverse_proxy 127.0.0.1:8080
}

director.example.com {
    reverse_proxy 127.0.0.1:3000
}
```

**3x-ui 设置要点**：

- REALITY 入站监听 `127.0.0.1:10000`，伪装域名（`serverName`）填真实的外部 HTTPS 站点（如 `www.microsoft.com`），**不要**填 `vless.example.com`。
- 面板监听 `127.0.0.1:32451`，SSL 证书**清空**，由 Caddy 负责 TLS。（由于我们这里把默认流量交给了 3x-ui，所以这里不用管，照旧）
- “入站”的基本配置里，“自定义分享地址”里填写 `vless.example.com` 或 `vless.example.com:443`，SNI 填伪装目标。

**优点**：`stream` 模块源安装即支持；Nginx 的 SNI 分流经过大量生产验证；Caddy 自动管理内部站点证书。
**缺点**：需要确认 Nginx 加载了 `stream` 和 `ssl_preread` 模块（多数发行版默认包含）。

### 方案二：Caddy L4 做前端入口

如果你更偏好 Caddy 生态，可以让 Caddy 用 `caddy-l4` 插件站在 443 做 SNI 分流，Nginx 退到内部做 HTTP 反代。但需要额外编译插件。

**端口分配**：

| 组件 | 监听地址 | 说明 |
|---|---|---|
| Caddy layer4 | `0.0.0.0:443` | 前端入口，需 caddy-l4 插件 |
| Caddy HTTP | `127.0.0.1:8443` | 内部 HTTPS |
| Nginx | `127.0.0.1:8080` | 内部 HTTP 站点 |
| 3x-ui | 同方案一 | 不变 |

**流量路径**：
```
用户: 443 → Caddy layer4 (SNI)
    ├─ SNI = vless.example.com → 127.0.0.1:10000 (3x-ui REALITY)
    ├─ SNI = example.com → 127.0.0.1:8443 (Nginx)
    ├─ SNI = www.example.com → 127.0.0.1:8443 (Nginx)
    ├─ SNI = m.example.com → 127.0.0.1:8443 (Nginx)
    ├─ SNI = api.example.com → 127.0.0.1:8443 (Nginx)
    ├─ SNI = site.example.com → 127.0.0.1:6443 (Caddy)
    ├─ SNI = director.example.com → 127.0.0.1:6443 (Caddy)
    └─ 其他 SNI → 127.0.0.1:10000 (3x-ui REALITY)
                    ├─ panel.example.com → 127.0.0.1:32451 (3x-ui 面板)
                    └─ sub.example.com  → 127.0.0.1:2096 (3x-ui 订阅服务)
```

**安装 caddy-l4**：

命令：

```bash
xcaddy build --with github.com/mholt/caddy-l4
```

或从 Caddy 官网下载页勾选 layer4 模块替换系统二进制。

**Caddyfile**：

```caddyfile
{
    layer4 {
        0.0.0.0:443 {
            @reality tls sni vless.example.com
            @nginx tls sni example.com
            @nginx tls sni www.example.com
            @nginx tls sni www.example.com
            @nginx tls sni m.example.com
            @nginx tls sni api.example.com
            @caddy tls sni site.example.com
            @caddy tls sni director.example.com

            route @reality {
                proxy 127.0.0.1:10000
            }
            route @nginx {
                proxy 127.0.0.1:8443
            }
            route @caddy {
                proxy 127.0.0.1:6443
            }
            # default
            route {
                proxy 127.0.0.1:10000
            }
        }
    }

    https_port 6443
}

site.example.com {
    reverse_proxy 127.0.0.1:8080
}

director.example.com {
    reverse_proxy 127.0.0.1:3000
}
```

**Nginx 配置**（只做内部 HTTP，不再碰 443）：

```nginx
server {
    listen 127.0.0.1:8443;
    server_name example.com *.example.com;
    location / {
        proxy_pass http://127.0.0.1:4000;
        proxy_set_header Host $host;
    }
}
```

**优点**：Caddyfile 更简洁，证书全自动。
**缺点**：需编译 caddy-l4；Caddy 的 L4 生态不如 Nginx stream 成熟。

### 方案对比与选择建议

| 对比项 | 方案一：Nginx Stream 入口 | 方案二：Caddy L4 入口 |
|---|---|---|
| **前端** | Nginx（stream 模块） | Caddy（caddy-l4 插件） |
| **内部 Web 引擎** | Caddy | Nginx |
| **额外依赖** | 无（源安装通常已支持） | 需编译 caddy-l4 |
| **REALITY 走 443** | ✅ SNI 匹配节点域名转本地 10000 | ✅ 同 |
| **稳定性** | 生产验证充分 | 可用，但插件生态较小 |
| **适用场景** | 优先推荐，追求稳定和可维护性 | 偏好 Caddy 语法，愿意折腾插件 |

**如果第一次部署，直接选方案一（Nginx Stream）**。它不需要额外编译任何东西，`stream` 模块的 SNI 分流是经过大量实践验证的方案，排错也相对容易。Caddy 在内部负责它最擅长的自动 HTTPS，Nginx 在前端只做它最擅长的 TCP 分流，3x-ui 老老实实待在本地后端——三者各司其职，互不干扰。
