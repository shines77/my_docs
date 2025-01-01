# Ubuntu 上如何查看分区信息

在 Ubuntu 上查看分区信息有多种方法，可以使用命令行工具或图形化工具。以下是常用的几种方法：

---

### 1. 使用 `df` 命令

`df` 命令可以查看已挂载文件系统的磁盘使用情况。
```bash
df -h
```
输出示例：
```
Filesystem      Size  Used Avail Use% Mounted on
/dev/sda1        50G   20G   28G  42% /
/dev/sda2        30G   10G   18G  36% /home
```
- `Filesystem`：分区设备名称。
- `Size`：分区总大小。
- `Used`：已用空间。
- `Avail`：可用空间。
- `Mounted on`：挂载点。

---

### 2. 使用 `fdisk` 命令

`fdisk` 是一个强大的分区管理工具，可以用来查看磁盘和分区信息。
```bash
sudo fdisk -l
```
输出示例：
```
Disk /dev/sda: 100 GiB, 107374182400 bytes, 209715200 sectors
Device     Boot   Start       End   Sectors  Size Id Type
/dev/sda1  *       2048 104857599 104855552   50G 83 Linux
/dev/sda2     104857600 167772159  62914560   30G 83 Linux
/dev/sda3     167772160 209715199  41943040   20G 82 Linux swap
```
- `Device`：分区设备名称。
- `Size`：分区大小。
- `Type`：分区类型。

---

### 3. 使用 `lsblk` 命令

`lsblk` 命令可以列出所有块设备（包括磁盘和分区）的树状结构。
```bash
lsblk
```
输出示例：
```
NAME   MAJ:MIN RM   SIZE RO TYPE MOUNTPOINT
sda      8:0    0   100G  0 disk
├─sda1   8:1    0    50G  0 part /
├─sda2   8:2    0    30G  0 part /home
└─sda3   8:3    0    20G  0 part [SWAP]
```
- `NAME`：设备名称。
- `SIZE`：设备或分区的大小。
- `TYPE`：类型（`disk` 表示磁盘，`part` 表示分区）。
- `MOUNTPOINT`：挂载点。

---

### 4. 使用 `parted` 命令

`parted` 是一个功能强大的分区工具，可以用来查看磁盘和分区信息。
```bash
sudo parted -l
```
输出示例：
```
Model: ATA VBOX HARDDISK (scsi)
Disk /dev/sda: 107GB
Partition Table: gpt
Number  Start   End    Size   File system  Name  Flags
 1      1049kB  53.7GB  53.7GB  ext4
 2      53.7GB  90.0GB  36.3GB  ext4
 3      90.0GB  107GB   17.0GB  linux-swap(v1)
```

---

### 5. 使用 `blkid` 命令

`blkid` 命令可以查看分区的 UUID 和文件系统类型。
```bash
sudo blkid
```
输出示例：
```
/dev/sda1: UUID="1234-5678-90AB" TYPE="ext4" PARTUUID="abcd1234"
/dev/sda2: UUID="5678-90AB-1234" TYPE="ext4" PARTUUID="efgh5678"
/dev/sda3: UUID="90AB-1234-5678" TYPE="swap" PARTUUID="ijkl9012"
```

---

### 6. 使用图形化工具 GParted

如果你更喜欢图形化界面，可以安装并使用 `GParted`：
1. 安装 GParted：
   ```bash
   sudo apt update
   sudo apt install gparted
   ```
2. 启动 GParted：
   ```bash
   sudo gparted
   ```
3. 在界面中选择磁盘，查看分区信息。

---

### 总结
- **快速查看分区信息**：使用 `lsblk` 或 `df`。
- **查看详细分区表**：使用 `fdisk` 或 `parted`。
- **查看 UUID 和文件系统类型**：使用 `blkid`。
- **图形化工具**：使用 `GParted`。

根据你的需求选择合适的方法即可！