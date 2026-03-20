# 如何使用 ZeroTier

## 1. 安装 ZeroTier

从 [zerotier.com/download](https://www.zerotier.com/download) 下载 ZeroTier，选择你相应系统的版本来安装。

> ZeroTier 支持当前的主要版本以及之前的两个主要版本。目前支持的版本包括 v1.16.x、v1.14.x 和 v1.12.x。

## 1.1 Windows

Windows 版下载 MSI Installer 安装包即可。

> 安装过程中请务必批准安装驱动。注意：Windows 7 和 Server 2012 用户请下载 [ZeroTier One 1.6.6](https://download.zerotier.com/RELEASES/1.6.6/dist/ZeroTier%20One.msi)，因为 ZeroTier One 1.8 及更高版本不支持 Windows 7。

## 1.2 Linux

基于 Debian 和 RPM 的发行版，包括 Debian、Ubuntu、CentOS、RHEL、Fedora 等，都通过脚本支持，脚本添加正确的仓库并安装包。其他 Linux 发行版可能有自己的包。如果没有，试着从源代码构建和安装。

如果你愿意依赖 SSL 来认证网站，只需一行命令安装即可：

```bash
curl -s https://install.zerotier.com | sudo bash
```

如果你安装了 GPG，还有更安全的选项：

```bash
curl -s 'https://raw.githubusercontent.com/zerotier/ZeroTierOne/main/doc/contact%40zerotier.com.gpg' | gpg --import && \
if z=$(curl -s 'https://install.zerotier.com/' | gpg); then echo "$z" | sudo bash; fi
```

使用脚本后，使用 apt 或 yum 管理未来对 zerotier-one 的更新。

## 2. ZeroTier管理后台

进入 zerotier 官网：[zerotier.com](https://zerotier.com)，注册一个账号，并登陆。

ZeroTier Central 有两个版本：

- New Central：现代化界面，功能增强，用户体验提升（[central.zerotier.com](https://central.zerotier.com)）
- Legacy Central：原始界面（[my.zerotier.com](https://my.zerotier.com)）

### 2.1 创建你的组织

进入 New Central：[https://central.zerotier.com/]()，先创建一个 `Organization` (组织)，例如：`i77studio`。

### 2.2 创建网络组

点击进入上面创建的组织，点击右上角的 `New Network Groups` 按钮，创建一个网络组，例如："my-network-group"。

### 2.3 创建网络

点击前面创建的`Network Groups`，点右侧的按钮 `New Network` 按钮，创建一个网络，例如："pensive-suess"。

### 2.4 加入你的网络

点击进入前面创建的 `Network`，复制右上角的 `Network ID`，例如：`e812ds922df921a0`，是一个 16 个字符的 16 进制字符串。

如果是 `Windows` 上，在右下角的图标 `zerotier` 图标上点右键，选择 `Join New Network`，填入 `Network ID` 即可。

如果是 Linux\MacOS 的 CLI，则使用如下命令：

```basg
sudo zerotier-cli join NETWORK_ID
```

### 2.5 授权您的设备

zerotier 网页端会要你确认授权，在通知弹窗中点击 `Authorized` 按钮，这是批量授权，即如果同时有多个授权请求，都会同意。

所以更推荐另一种方式，即 “直接管理设备”，上面那个通知弹窗可以先点 `Reject` (拒绝)。

然后在 `Member Devices` 的列表里，最右边有个 `Action`，每一个设备都有个 `...` 的按钮，点击即可弹出授权，取消授权和拒绝的选项。

> 建议直接管理设备以避免意外的大规模授权。

## 4. 搭建 ZeroTier Moon 服务器

官方 Moon 中转服务器在国外，国内客户端使用延迟大，甚至出现访问不了的问题。可以自己搭建 Moon 中转服务器，来实现稳定的服务。Moon 服务器需要一个静态公网 IP。

## 4.1 配置 Moon 服务器

进入 ZeroTier 配置文件目录。

```bash
cd /var/lib/zerotier-one
````

生成 moon.json 文件。

```bash
zerotier-idtool initmoon identity.public >>moon.json
```

编辑 moon.json 文件。

找到 `"stableEndpoints": []`，添加："IPv4地址/9993" 或者 "IPv4地址/9993","IPv6地址/9993" 。

例如：

```bash
"stableEndpoints": ["114.114.114.114/9993"]
```

一定要用 9993 端口，因为这是 Moon 服务器默认使用的端口。

注意：记录下 moon.json 文件中的 id，这个 ID 也就是你的 Address ID，也要在 Moon 中使用。

生成 .moon 签名文件：

```bash
zerotier-idtool genmoon moon.json
```

这将生成一个名为 {Your_Address_ID}.moon 的文件，例如：`0000006eadbeef00.moon`。

创建 moon 结点文件夹：

```bash
mkdir /var/lib/zerotier-one/moons.d
```

将签名文件复制到 moons.d 文件夹中：

```bash
cp {Your_Address_ID}.moon ./moons.d/
```

最后，重启 ZeroTier 服务让其生效：

```bash
/etc/init.d/zerotier-one restart
```

## 5. 卸载 ZeroTier

**Ubuntu / Debian 卸载方法**

通过 dpkg 删除 zerotier-one 服务：

```bash
sudo dpkg -P zerotier-one
```

删除 zerotier-one 文件夹，该文件夹存储了 address 地址，删除后再次安装会获得新的 address 地址。

```bash
sudo rm -rf /var/lib/zerotier-one/
```

**Windows 卸载**

Windows卸载

在“应用和功能”中卸载 ZeroTier One 。

直接卸载之后，还需要手工删除一些残留的文件。

打开 C 盘，删除残留文件：

1：`C:\Users\用户名\AppData\Local` 中的 zerotier 文件

2：`C:\Program Files` 中的 zerotier 文件

3：`C:\ProgramData` 中的 zerotier 文件

4：`C:\Program Files (x86)` 中的 zerotier 文件

## X. 参考

- [ZeroTier Docs: Quickstart Guide](https://docs.zerotier.com/quickstart/)

- [使用ZeroTier实现内网穿透并异地组网](https://www.cnblogs.com/ubirdy/p/18721780)
