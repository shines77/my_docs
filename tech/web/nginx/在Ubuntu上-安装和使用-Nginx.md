# 在 Ubuntu 上安装和使用 Nginx

## 1. apt-get安装

```bash
# 切换至 root 用户
sudo su root
apt update

# 安装 Nginx
apt-get install nginx

# 查看是否安装成功
nginx -v

# 启动 Nginx
sudo service nginx start
或
sudo systemctl start nginx

# 检查 Nginx 的状态
sudo systemctl status nginx

# 设置开机启动
sudo systemctl enable nginx

# 更改设置后, 测试配置
sudo nginx -t
# 重新加载配置
sudo systemctl reload nginx
```

Nginx 安装完成后文件位置：

- `/usr/sbin/nginx`：主程序
- `/etc/nginx`：存放配置文件
- `/usr/share/nginx`：存放静态文件
- `/var/log/nginx`：存放日志

如果你需要对 Nginx 进行配置，比如设置虚拟主机、修改默认端口等，你可以编辑 Nginx 的配置文件。这些文件通常位于 `/etc/nginx/` 目录下。

主配置文件：`/etc/nginx/nginx.conf` 。

虚拟主机配置文件：通常位于 `/etc/nginx/sites-available/` 目录下，并通过符号链接到 `/etc/nginx/sites-enabled/` 目录来启用。

在修改任何配置文件之前，建议先备份原始文件。

## 2. 使用 Nginx 官方仓库

如果你想要安装 Nginx 的最新版本，你可以添加 Nginx 的官方仓库。

### 2.1 导入 Nginx 签名密钥

```bash
sudo mkdir /etc/apt/keyrings
curl -fsSL https://nginx.org/keys/nginx_signing.key | sudo gpg --dearmor -o /etc/apt/keyrings/nginx-archive-keyring.gpg
```

### 2.2 添加 Nginx 仓库

```bash
echo "deb [signed-by=/etc/apt/keyrings/nginx-archive-keyring.gpg] http://nginx.org/packages/ubuntu `lsb_release -cs` nginx" | sudo tee /etc/apt/sources.list.d/nginx.list > /dev/null
```

### 2.3 ‌更新并安装 Nginx

```bash
sudo apt update
sudo apt install nginx
```

## 3. 使用 snap 安装

如果你不想使用仓库，也可以使用 Snap 包管理工具来安装 Nginx ：

```bash
sudo snap install nginx
```

## 4. 编译安装

### 4.1 卸载已安装的 Nginx

```bash
# 彻底卸载 Nginx
apt-get --purge autoremove nginx

# 查看 Nginx 的版本号
nginx -v
```

### 4.2 安装依赖包

```bash
apt-get install gcc
apt-get install libpcre3 libpcre3-dev
apt-get install zlib1g zlib1g-dev

# Ubuntu 14.04 的仓库中没有发现 openssl-dev，由下面 openssl 和 libssl-dev 替代
# apt-get install openssl openssl-dev
sudo apt-get install openssl 
sudo apt-get install libssl-dev
```

注：如果系统源没有 zlib 的安装包，可能还需要自己手动编译 zlib 源码，在这里：[https://zlib.net](https://zlib.net) 下载。

### 4.3 安装 Nginx

源码下载到你的用户目录 `/home/{$your_username}` 下：

```bash
cd /home/{$your_username}
mkdir nginx
cd nginx

wget https://nginx.org/download/nginx-1.28.2.tar.gz
tar -xvf nginx-1.28.2.tar.gz
```

### 4.4 编译 Nginx

编译和安装到 `/usr/local/nginx` 目录：

```bash
# 进入 Nginx 目录
cd ./nginx-1.28.2

# 执行命令
./configure --prefix=/usr/local/nginx

# 执行 make 命令, 如果你的 CPU 核心多，把 2 改为你的 CPU 核心数
make -j2

# 执行 make install 命令
make install
```

### 4.5 启动 Nginx

```bash
# 进入 Nginx 启动目录
cd /usr/local/nginx/sbin

# 启动 Nginx
./nginx

# 快速停止
./nginx -s stop
# 优雅关闭，在退出前完成已经接受的连接请求
./nginx -s quit
# 重新加载配置
./nginx -s reload
```

## 5. 把 Nginx 安装成系统服务

### 5.1 创建服务脚本

```bash
vim /usr/lib/systemd/system/nginx.service
```

脚本内容：（注：在脚本中替换成自己的 Nginx 安装目录）

```bash
[Unit]
Description=nginx - web server
After=network.target remote-fs.target nss-lookup.target

[Service]
Type=forking
PIDFile=/usr/local/nginx/logs/nginx.pid
ExecStartPre=/usr/local/nginx/sbin/nginx -t -c /usr/local/nginx/conf/nginx.conf
ExecStart=/usr/local/nginx/sbin/nginx -c /usr/local/nginx/conf/nginx.conf
ExecReload=/usr/local/nginx/sbin/nginx -s reload
ExecStop=/usr/local/nginx/sbin/nginx -s stop
ExecQuit=/usr/local/nginx/sbin/nginx -s quit
PrivateTmp=true

[Install]
WantedBy=multi-user.target
````

重新加载系统服务：

```bash
sudo systemctl daemon-reload
```

启动服务：

```bash
sudo systemctl start nginx.service
```

可设置 Nginx 开机启动：

```bash
sudo systemctl enable nginx.service
```

检查 Nginx 是否启动：

```bash
ps -ef | grep nginx
# 或者
sudo systemctl status nginx
```

## 6. 配置防火墙（如果需要）

如果你在使用 UFW（Uncomplicated Firewall），确保开放 HTTP 和 HTTPS 端口：

```bash
sudo ufw allow 'Nginx HTTP'
sudo ufw allow 'Nginx HTTPS'
```

关闭防火墙：

```bash
sudo systemctl stop firewalld.service
```

禁止防火墙开机启动：

```bash
sudo systemctl disable firewalld.service
```

放行 80, 433 端口：

```bash
sudo firewall-cmd --zone=public --add-port=80/tcp --permanent
sudo firewall-cmd --zone=public --add-port=433/tcp --permanent
```

重启防火墙：

```bash
sudo firewall-cmd --reload
```

## X. 参考文章

- [Ubuntu 14.04 安装 nginx](https://developer.aliyun.com/article/1369201)

- [Ubuntu 24.04 源码安装 nginx](https://blog.csdn.net/xrinosvip/article/details/144326935)
