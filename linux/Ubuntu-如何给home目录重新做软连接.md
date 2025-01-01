# Ubuntu 如何给 /home 目录重新做软连接

是的，在 Ubuntu 上可以将 `/home` 目录软链接到其他分区（例如 `/userdata`），但需要注意一些细节和步骤，以确保系统正常运行。以下是具体操作步骤：

---

### 1. 准备新分区

假设你已经有一个分区（例如 `/dev/sdb1`），并打算将其挂载到 `/userdata`，然后将 `/home` 链接到 `/userdata/home`。

1. **格式化分区**（如果需要）：

   ```bash
   sudo mkfs.ext4 /dev/sdb1
   ```

2. **创建挂载点**：

   ```bash
   sudo mkdir /userdata
   ```

3. **挂载分区**：

   ```bash
   sudo mount /dev/sdb1 /userdata
   ```

4. **检查挂载**：

   ```bash
   df -h
   ```

   确保 `/dev/sdb1` 已正确挂载到 `/userdata`。

---

### 2. 迁移 `/home` 数据

将现有的 `/home` 目录内容复制到新分区：

```bash
sudo rsync -av /home/ /userdata/home/
```

- `-a`：保留文件属性和权限。
- `-v`：显示详细过程。

---

### 3. 备份并删除原 `/home` 目录

1. **备份原 `/home` 目录**（可选但建议）：

   ```bash
   sudo mv /home /home.backup
   ```

2. **删除原 `/home` 目录**（这里其实不用再删除，因为 mv 是改名）：

   ```bash
   sudo rm -rf /home
   ```

---

### 4. 创建软链接

将 `/home` 链接到 `/userdata/home`：

```bash
sudo ln -s /userdata/home /home
```

---

### 5. 验证软链接

检查软链接是否创建成功：

```bash
ls -l /home
```

输出应类似于：

```
lrwxrwxrwx 1 root root 12 Oct  1 12:34 /home -> /userdata/home
```

---

### 6. 设置开机自动挂载

为了确保系统重启后分区自动挂载，需要编辑 `/etc/fstab` 文件：
1. 获取分区的 UUID：

   ```bash
   sudo blkid /dev/sdb1
   ```

   输出示例：

   ```
   /dev/sdb1: UUID="1234-5678-90AB" TYPE="ext4"
   ```

2. 编辑 `/etc/fstab`：

   ```bash
   sudo nano /etc/fstab
   ```

3. 添加以下内容：

   ```
   UUID=1234-5678-90AB  /userdata  ext4  defaults  0  2
   ```

4. 保存并退出（`Ctrl+O` 保存，`Ctrl+X` 退出）。

---

### 7. 重启系统

重启系统以验证配置是否生效：

```bash
sudo reboot
```

重启后检查 `/home` 是否正常链接到 `/userdata/home`。

---

### 注意事项

1. **备份数据**：操作前务必备份重要数据，以防意外。
2. **权限问题**：确保 `/userdata/home` 的权限和所有者与原来的 `/home` 一致。
3. **多用户系统**：如果系统有多个用户，确保每个用户的主目录权限正确。
4. **软链接 vs 挂载**：如果希望直接将分区挂载到 `/home`，可以跳过软链接步骤，直接在 `/etc/fstab` 中将分区挂载到 `/home`。

---

通过以上步骤，你可以成功将 `/home` 目录软链接到其他分区（如 `/userdata`），从而扩展存储空间或优化磁盘管理。
