# 如何使用 ZeroTier 科学上网？

## :one: 概述

使用公网服务器配合 ZeroTier 实现科学上网，核心思路是将其配置为 VPN 出口节点（Exit Node）。所有流量会先通过 ZeroTier 网络汇聚到这台服务器上，再借助服务器的公网 IP 进行访问。

整个过程分为三个步骤：

1. 第一步：服务器配置。安装 ZeroTier，开启 IP 转发，配置 iptables NAT 规则。

2. 第二步：ZeroTier 后台网络配置。进入 ZeroTier 官网管理后台，添加路由规则，添加服务器和各个 ZeroTier 客户端的授权。由 ZeroTier 后台把路由规则下发到虚拟局域网内的所有设备。

3. 第三步：客户端配置。安装 ZeroTier，加入网络，开启允许默认路由，这样就 OK 了。

## 📡 第一步：在公网服务器上部署

首先，你需要登录到你的公网服务器，执行以下操作，让它有能力转发网络流量。

### 1. 安装并加入网络

在服务器上安装 ZeroTier 客户端，并让它加入你在官网创建的网络。

```bash
# 下载安装脚本
curl -s https://install.zerotier.com | sudo bash

# 加入网络
sudo zerotier-cli join [你的网络ID]
```

### 2. 开启IP转发

这是让服务器从一个“客户端”变成“路由器”的关键一步。编辑系统配置文件并使其生效。

此部分可以参考官网的文档：[Route between ZeroTier and Physical Networks](https://docs.zerotier.com/route-between-phys-and-virt/)

```bash
# 编辑配置文件
echo 'net.ipv4.ip_forward=1' | sudo tee -a /etc/sysctl.conf

# 使配置立即生效
sudo sysctl -p
```

### 3. 配置 iptables

通过 iptables 规则，配置网络地址转换 (NAT)，让服务器为其身后的设备（你的手机、电脑）做网络地址转换，从而实现上网功能。

使用 `ifconfig` 命令查看你的服务器上网的网卡名称叫什么，物理机一般叫 `eth0`，VPS 云服务器有可能叫 `ens17` 之类的。

并找到你要转发的 ZerTier 虚拟网卡叫什么，例如：ztugaqveyu 。

也可以使用下列命令：

```bash
# 先确认你的网卡名称，一般外网网卡是 eth0 或 ens3，ZeroTier 虚拟网卡以 zt 开头
ifconfig

# 或者
ip link show
```

如何判断哪个是上网的网卡，也可以使用 `route -n` 命令，其中 Destination 为 `0.0.0.0`，同时也有 Gateway 的就是上网的网卡了。例如：

```bash
Kernel IP routing table
Destination     Gateway         Genmask         Flags Metric Ref    Use Iface
0.0.0.0         123.123.111.1   0.0.0.0         UG    0      0        0 ens17
10.0.0.0        0.0.0.0         255.0.0.0       U     0      0        0 ens18
10.114.100.0    0.0.0.0         255.255.255.0   U     0      0        0 ztfl6dcgxb
10.144.172.0    0.0.0.0         255.255.255.0   U     0      0        0 ztugaqveyu
123.123.111.0   0.0.0.0         255.255.255.0   U     0      0        0 ens17
```

先安装 iptables 配置持久化：

```bash
# 安装持久化工具，让重启后规则依然生效（以 Debian/ Ubuntu 为例）
sudo apt-get install iptables-persistent
```

设置一些 Shell 变量（并对其进行个性化设置）：

```bash
ETH0_IFACE=ens17; ZT_IFACE=ztugaqveyu
```

添加 iptables 规则：

```bash
# 配置 NAT 规则
sudo iptables -t nat -A POSTROUTING -o $ETH0_IFACE -j MASQUERADE

#（注意！与下一条只能二选一，不要重复了）
# 允许从公网接口向 ZeroTier 接口转发已建立的连接（该命令出自官方文档）
sudo iptables -A FORWARD -i $ETH0_IFACE -o $ZT_IFACE -m state --state RELATED,ESTABLISHED -j ACCEPT

#（注意！与上一条只能二选一，不要重复了）
# 允许转发已建立的连接（该命令出自 DeepSeek-V3.2）
sudo iptables -A FORWARD -m conntrack --ctstate RELATED,ESTABLISHED -j ACCEPT

# 允许从 ZeroTier 接口向公网接口转发流量
sudo iptables -A FORWARD -i $ZT_IFACE -o $ETH0_IFACE -j ACCEPT
```

使用下列命令查看添加后的结果：

```bash
# 查看 NAT 的转发，默认 POSTROUTING 就是全部转发的，所以设置改了也看不出来
iptables -t nat -L

# 查看流量转发
iptables -S
```

然后保存配置持久化，重启以后依然有效：

```bash
sudo netfilter-persistent save
```

> .
> 注意：如果你用的是 CentOS/RHEL 系统，执行 `sudo yum install iptables-services`，然后用 `sudo service iptables save` 保存规则。
> .

可以查看下列文件，如果有重复添加的，可以手动删除一下：

```bash
vim /etc/iptables/rules.v4
vim /etc/iptables/rules.v6
```

## 🎛 第二步：在ZeroTier官网后台设置路由

服务器配置好后，需要告诉 ZeroTier 的网络控制器，这台服务器就是“大门”。

1. 登录 [ZeroTier Legacy Central](https://my.zerotier.com/)，进入你的网络管理页面。

2. 在 Members 列表中找到你的服务器，勾选 Auth 复选框，授权它加入网络。

3. 点击顶部导航栏的 Settings，找到 Managed Routes 部分，点击 Add Route 添加一条新路由：

   - Destination (目标网络)：填写 0.0.0.0/0（这代表所有互联网流量）。

   - Via (经由)：填写你的服务器在 ZeroTier 网络中被分配的 虚拟 IP 地址（例如：10.114.100.1）。

   - 为了打通你的其他设备的内部局域网网段，还可以添加互相访问的配置路由表。

      - 例如：`192.168.3.0/24 via 10.114.100.66`，`192.168.5.0/24 via 10.114.100.77` 。

4. 保存即可。

注：由于 `New Central` (现代化界面) 免费版无法配置路由和 DNS，所以这里只能选择 `Legacy Central`，否则至少要升级到基础班，每个月需要 18 美元。

## 📱 第三步：配置你的客户端设备

最后一步，让你的电脑或手机把流量“甩”给刚才设置好的服务器。

1. 在你想要科学上网的设备上（如 Windows、macOS、Android、iOS）安装 ZeroTier 客户端。

2. 打开客户端，加入你创建的那个 网络ID。

3. 回到 ZeroTier 官网后台，在 Members 列表中找到你的设备，同样勾选 Auth 授权。

4. 在客户端上进行最后的“一键切换”：

   - Windows/Linux/MacOS：右键点击系统托盘/菜单栏的 ZeroTier 图标，找到你的网络，勾选 Allow Default Route（允许默认路由）或类似选项。

   - Android/iOS：在 ZeroTier App 里，找到你的网络，开启 Route All Traffic（路由所有流量）的开关。

## 💡 进阶优化建议

**自建 Moon 节点提高稳定性**：ZeroTier 官方服务器在国外，有时打洞会慢。你可以直接用这台境外服务器搭建一个 Moon 节点，这会极大地加快设备间的连接速度，延迟直降 70% 以上。简单说就是让服务器充当一个“加速握手”的角色，但流量依然是直连的（如果成功打洞）或通过此服务器转发。

**启用 IPv6**：如果你的服务器支持 IPv6，强烈建议配置。使用 IPv6 可以完全避免 NAT 的复杂性和潜在问题，让连接更顺畅，并且不影响你访问 IPv6 资源。

**解决 Linux 客户端的 rp_filter 问题**：如果你的客户端是 Linux 系统，开启 allowDefault 后发现无法上网，可以尝试执行以下命令关闭反向路径过滤：

```bash
# 编辑配置文件
echo 'net.ipv4.conf.all.rp_filter=2' | sudo tee -a /etc/sysctl.conf

# 使配置立即生效
sudo sysctl -p
```

也可以自己手动添加加到 `/etc/sysctl.conf` 文件里。

配置完成后，你可以在手机上关闭 Wi-Fi，用 4G/5G 网络测试一下，访问 [https://ip.sb](https://ip.sb) 这类网站，显示的应该就是你公网服务器的 IP 地址了。

## 其他

Windows 10 修改网络配置的 `公用网络`、`专业网络` 属性。

Windows PowerShell (需要管理员权限)。

按网络名称修改（推荐）：

```powershell
Set-NetConnectionProfile -Name "网络 10" -NetworkCategory Private
```

按接口别名修改：

```powershell
Set-NetConnectionProfile -InterfaceAlias "以太网 2" -NetworkCategory Private
```

## :book: 参考文章

- [DeepSeek-V3.2 问答](https://chat.deepseek.com/a/chat/s/0486ff7c-29bb-420f-b63a-0fbdd38e3929)

- [NAS内网穿透方案之ZeroTier方案](https://zhuanlan.zhihu.com/p/710321822)
