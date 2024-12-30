# Ubuntu 中如何让 DNS 一直有效

在 Linux 系统中，`/etc/resolv.conf` 文件用于配置 DNS 服务器。为了防止该文件被自动覆盖或重置，可以采取以下措施：

### 1. 设置文件为不可变

使用 `chattr` 命令将文件设为不可变，防止其被修改。

```bash
sudo chattr +i /etc/resolv.conf
```

**取消不可变：**

```bash
sudo chattr -i /etc/resolv.conf
```

### 2. 禁用自动生成

某些系统（如使用 `systemd-resolved` 或 `NetworkManager`）会自动生成 `/etc/resolv.conf`，可以通过以下方式禁用：

#### 对于 `systemd-resolved`：
编辑 `/etc/systemd/resolved.conf`，设置：

```ini
[Resolve]
DNS=your_dns_server
FallbackDNS=
#Domains=
#LLMNR=no
#MulticastDNS=no
#DNSSEC=no
#Cache=no
#DNSStubListener=no
```

然后重启服务：

```bash
sudo systemctl restart systemd-resolved
```

#### 对于 `NetworkManager`：

编辑 `/etc/NetworkManager/NetworkManager.conf`，在 `[main]` 部分添加：

```ini
[main]
dns=none
```

然后重启服务：

```bash
sudo systemctl restart NetworkManager
```

### 3. 手动配置

直接编辑 `/etc/resolv.conf`，添加 DNS 服务器：

```bash
nameserver 8.8.8.8
nameserver 8.8.4.4
```

### 4. 使用 `resolvconf` 工具

如果系统安装了 `resolvconf`，可以通过以下步骤配置：

1. 编辑 `/etc/resolvconf/resolv.conf.d/head`，添加 DNS 服务器：

    ```bash
    nameserver 8.8.8.8
    nameserver 8.8.4.4
    ```

2. 更新配置：

    ```bash
    sudo resolvconf -u
    ```

### 5. 使用 `netplan`（适用于 Ubuntu）

如果使用 `netplan`，编辑 `/etc/netplan/*.yaml` 文件，添加 DNS 配置：

```yaml
network:
  version: 2
  ethernets:
    eth0:
      dhcp4: no
      addresses:
        - 192.168.1.100/24
      gateway4: 192.168.1.1
      nameservers:
        addresses:
          - 8.8.8.8
          - 8.8.4.4
```

然后应用配置：

```bash
sudo netplan apply
```

### 总结

根据系统环境选择合适的方法，确保 `/etc/resolv.conf` 的 DNS 配置不会被覆盖。
