## 问题

nginx 的 HTTP 重定向到 HTTPS 怎么把子域名也带进去？

## 回答

（DeepSeek v4.0）

在 Nginx 里做 HTTP → HTTPS 重定向时，想**保留原来的子域名**，关键是不要写死目标域名，而是用变量 `$host`（或 `$server_name`）。下面分几种常见写法。

## 一、最通用的写法：用 `$host` 变量

`$host` 会取请求里的主机名（优先 Host 头，其次请求行，最后匹配的 server_name），所以不管访问的是 `www.xxx.com` 还是 `api.xxx.com`，都会跳到对应的 HTTPS 地址。

```nginx
server {
    listen 80;
    listen [::]:80;
    server_name _;              # 匹配所有域名，包括所有子域名

    return 301 https://$host$request_uri;
}
```

- `$host`：自动带上请求的子域名
- `$request_uri`：带上原始路径和查询参数（如 `/a/b?x=1`）
- `server_name _;`：兜底匹配所有域名

这样：
- `http://www.example.com/a` → `https://www.example.com/a`
- `http://api.example.com/b?x=1` → `https://api.example.com/b?x=1`

## 二、只想对特定几个域名重定向

如果你只希望某些域名跳转，明确列出即可：

```nginx
server {
    listen 80;
    server_name example.com www.example.com api.example.com;

    return 301 https://$host$request_uri;
}
```

## 三、通配符匹配所有子域名

```nginx
server {
    listen 80;
    server_name example.com *.example.com;

    return 301 https://$host$request_uri;
}
```

`*.example.com` 匹配一级子域名（如 `www`、`api`），但**不匹配多级**（如 `a.b.example.com`）。要匹配多级可以用正则：

```nginx
server {
    listen 80;
    server_name ~^(?<sub>.+)\.example\.com$ example.com;

    return 301 https://$host$request_uri;
}
```

## 四、`$host` vs `$server_name` vs `$http_host`

| 变量 | 含义 | 适用场景 |
|------|------|----------|
| `$host` | 请求的主机名（不含端口） | **推荐**，通用重定向 |
| `$server_name` | 当前 server 块匹配到的名字 | 固定单域名时用，通配符场景会不准 |
| `$http_host` | 客户端 Host 头原样（含端口） | 需要保留端口时用 |

**注意**：用 `$host` 时不会带端口；如果原请求是 `http://example.com:8080`，重定向后变成 `https://example.com/...`（443 默认端口），通常这正是你想要的。

如果确实要保留原端口（少见），可以用：

```nginx
return 301 https://$http_host$request_uri;
```

## 五、完整的 HTTPS 站点配合

重定向只是 HTTP 侧的事，HTTPS 站点还要单独配置，并且要覆盖同样的域名：

```nginx
# HTTP 重定向
server {
    listen 80;
    listen [::]:80;
    server_name example.com *.example.com;
    return 301 https://$host$request_uri;
}

# HTTPS 主站
server {
    listen 443 ssl;
    listen [::]:443 ssl;
    server_name example.com *.example.com;

    ssl_certificate     /path/fullchain.pem;
    ssl_certificate_key /path/privkey.pem;

    root /var/www/html;
    index index.html;
}
```

## 六、常见坑

1. **证书必须覆盖子域名**：如果 HTTPS 站点要接受 `api.example.com`，证书里必须包含该子域名（用通配符证书 `*.example.com` 或多域名证书），否则浏览器会报证书错误。
2. **`server_name _;` 会抢匹配**：如果同时存在精确匹配的 server 块，Nginx 会优先用精确的。用 `_` 做兜底重定向时，确保它是默认 server（加 `default_server`）。
3. **避免重定向循环**：不要在 443 的 server 块里又写 `return 301 https://...`，否则会无限跳转。
4. **HSTS**：确认所有子域名都能正常 HTTPS 后，再加 `add_header Strict-Transport-Security ...`，否则加了 HSTS 后子域名访问会出问题。

一句话总结：**HTTP 重定向用 `return 301 https://$host$request_uri;`，配 `server_name` 覆盖你要支持的域名和子域名即可。**