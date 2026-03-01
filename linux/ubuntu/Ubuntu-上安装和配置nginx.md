# Ubuntu 上安装和配置 nginx

在Ubuntu上安装和使用Nginx非常简单。Nginx是一款高性能的Web服务器和反向代理服务器，也常用作负载均衡器和HTTP缓存。

以下是完整的安装和基本使用指南：

---

## 📦 一、安装Nginx

### **方法一：使用Ubuntu官方仓库（简单稳定）**

```bash
# 1. 更新包索引
sudo apt update

# 2. 安装Nginx
sudo apt install nginx -y

# 3. 验证安装
nginx -v
```

### **方法二：使用Nginx官方仓库（版本更新）**

如果你想安装最新版Nginx，可以添加官方仓库：

```bash
# 1. 安装必要的依赖
sudo apt install curl gnupg2 ca-certificates lsb-release ubuntu-keyring -y

# 2. 导入官方Nginx签名密钥
curl https://nginx.org/keys/nginx_signing.key | gpg --dearmor | sudo tee /usr/share/keyrings/nginx-archive-keyring.gpg >/dev/null

# 3. 添加Nginx稳定版仓库
echo "deb [signed-by=/usr/share/keyrings/nginx-archive-keyring.gpg] http://nginx.org/packages/ubuntu `lsb_release -cs` nginx" | sudo tee /etc/apt/sources.list.d/nginx.list

# 4. 安装Nginx
sudo apt update
sudo apt install nginx -y
```

---

## 🚀 二、启动和管理Nginx

### **基本服务管理命令**

```bash
# 启动Nginx
sudo systemctl start nginx

# 设置开机自启
sudo systemctl enable nginx

# 查看运行状态
sudo systemctl status nginx

# 停止Nginx
sudo systemctl stop nginx

# 重启Nginx（完全重启）
sudo systemctl restart nginx

# 重载配置（不中断服务，平滑重载）
sudo systemctl reload nginx
```

### **验证Nginx是否运行**

```bash
# 方法1：查看进程
ps aux | grep nginx

# 方法2：测试端口（默认80端口）
sudo ss -tlnp | grep 80

# 方法3：浏览器访问
# 打开浏览器，访问 http://你的服务器IP 或 http://localhost
# 如果看到"Welcome to nginx"页面，说明安装成功
```

---

## ⚙️ 三、Nginx配置文件结构

### **主要目录和文件**

```
/etc/nginx/
├── nginx.conf          # 主配置文件
├── sites-available/     # 可用的站点配置
├── sites-enabled/       # 已启用的站点配置（符号链接到sites-available）
├── conf.d/              # 额外的配置文件
├── modules-available/   # 可用的模块
├── modules-enabled/     # 已启用的模块
└── snippets/            # 可重用的配置片段
```

### **默认网站根目录**

```bash
# 默认的网页文件存放位置
/var/www/html/

# 默认的欢迎页面
/var/www/html/index.nginx-debian.html
```

---

## 📝 四、配置一个简单的静态网站

### **步骤1：创建网站目录和文件**

```bash
# 创建网站目录
sudo mkdir -p /var/www/mysite

# 创建测试页面
echo "<h1>Hello from My Site!</h1>" | sudo tee /var/www/mysite/index.html

# 设置权限
sudo chown -R www-data:www-data /var/www/mysite
```

### **步骤2：创建站点配置文件**

```bash
# 在sites-available中创建配置文件
sudo nano /etc/nginx/sites-available/mysite
```

添加以下内容：

```nginx
server {
    listen 80;
    server_name mysite.local;  # 替换为你的域名或IP

    root /var/www/mysite;
    index index.html;

    location / {
        try_files $uri $uri/ =404;
    }

    # 可选：错误页面配置
    error_page 404 /404.html;
    error_page 500 502 503 504 /50x.html;
}
```

### **步骤3：启用站点并测试**

```bash
# 创建符号链接到sites-enabled
sudo ln -s /etc/nginx/sites-available/mysite /etc/nginx/sites-enabled/

# 测试配置文件语法
sudo nginx -t

# 如果显示"test is successful"，重载Nginx
sudo systemctl reload nginx
```

### **步骤4：配置本地hosts测试（可选）**

如果你在本地测试，可以修改hosts文件：

```bash
sudo nano /etc/hosts
# 添加一行：127.0.0.1 mysite.local
```

---

## 🔧 五、常见配置示例

### **1. 配置为反向代理（Node.js应用）**

```nginx
server {
    listen 80;
    server_name api.example.com;

    location / {
        proxy_pass http://localhost:3000;  # 转发到Node.js应用
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }
}
```

### **2. 配置SSL/HTTPS（使用Let's Encrypt）**

```bash
# 安装Certbot
sudo apt install certbot python3-certbot-nginx -y

# 自动获取并配置SSL证书
sudo certbot --nginx -d example.com -d www.example.com
```

### **3. 配置负载均衡**

```nginx
upstream backend {
    server 192.168.1.10:3000 weight=3;  # weight表示权重
    server 192.168.1.11:3000;
    server 192.168.1.12:3000 backup;    # 备份服务器
}

server {
    listen 80;
    server_name loadbalancer.example.com;

    location / {
        proxy_pass http://backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

---

## 📊 六、查看日志和调试

```bash
# 访问日志
sudo tail -f /var/log/nginx/access.log

# 错误日志
sudo tail -f /var/log/nginx/error.log

# 测试配置
sudo nginx -t

# 查看详细配置信息
sudo nginx -T
```

---

## 🔒 七、基本安全建议

1. **删除默认站点**（如果不用）：
   ```bash
   sudo rm /etc/nginx/sites-enabled/default
   ```

2. **隐藏Nginx版本号**：
   在 `/etc/nginx/nginx.conf` 的http块中添加：
   ```nginx
   server_tokens off;
   ```

3. **设置适当的权限**：
   ```bash
   sudo chown -R www-data:www-data /var/www/
   ```

4. **配置防火墙**：
   ```bash
   sudo ufw allow 80/tcp
   sudo ufw allow 443/tcp
   sudo ufw reload
   ```

---

## ✅ 八、常用命令速查表

| 命令 | 说明 |
|------|------|
| `sudo systemctl start nginx` | 启动Nginx |
| `sudo systemctl stop nginx` | 停止Nginx |
| `sudo systemctl restart nginx` | 重启Nginx |
| `sudo systemctl reload nginx` | 平滑重载配置 |
| `sudo systemctl status nginx` | 查看状态 |
| `sudo nginx -t` | 测试配置语法 |
| `sudo nginx -s reload` | 重载配置（另一种方式） |
| `sudo journalctl -u nginx` | 查看系统日志 |

---

现在你的Nginx应该已经安装并运行起来了！如果遇到任何问题，可以检查以下几点：
- 防火墙是否开放了80/443端口
- 配置文件语法是否正确（`nginx -t`）
- 日志文件中有没有错误信息

如果有具体的配置需求或遇到问题，欢迎继续问我！
