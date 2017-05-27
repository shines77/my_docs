
解决 Linux 服务器 nfs 挂载目录磁盘已满的问题
====================================================

## 1. 需求 ##

我们的一台阿里云服务器，这里简称 `文件服务器`，由于一开始的疏忽，上传的文件已经把系统盘 `40G` 占满了，几乎没空闲磁盘空间了。这台机器上原本就准备了一个`200G` 的磁盘，只是没有挂载使用而已（硬件上已经启用了的），它跟 `40G` 的系统盘一样，也是一个虚拟云盘。我们需要做的是，把系统盘上的一些文件或文件夹迁移到那未使用的 `200G` 数据盘上，以缓解系统盘的存储容量压力。

系统架构是这样的，有 `Web1`, `Web2`, `Web3`, `Web4` 共四台 `Web服务器` 通过 `nfs` 挂载的方式，都把本地的 `/var/lib/webfiles` 文件夹映射到了后端的这台 `文件服务器` 上的同名文件夹上，挂载过程如下：

    // 安装 nfs
    # apt-get install nfs-common

    // 挂载命令
    # mount 192.168.3.177:/var/lib/webfiles /var/lib/webfiles
    // 或者
    # mount 192.168.3.177:/data/var/lib/webfiles /var/lib/webfiles

    // 卸载 /var/lib/webfiles 的挂载
    # umount /var/lib/webfiles

（注：上面 `mount` 命令中也可以加入 `-t nfs` 参数试试，但实际使用中好像加这个参数反而会 `mount` 失败，不会报错，但挂载不成功。）

这里，假设 `192.168.3.177` 是那个 `文件服务器` 的 `IP地址` 。

## 2. 解决步骤 ##

### 2.1 磁盘设备名 ###

首先，登录到这台 `文件服务器` ，查看当前磁盘分区和挂载的情况：

    $ df -h

    Filesystem      Size  Used Avail Use% Mounted on
    udev            7.9G  4.0K  7.9G   1% /dev
    tmpfs           1.6G  420K  1.6G   1% /run
    /dev/vda1        40G   37G  354M 100% /
    none            4.0K     0  4.0K   0% /sys/fs/cgroup
    none            5.0M     0  5.0M   0% /run/lock
    none            7.9G     0  7.9G   0% /run/shm
    none            100M     0  100M   0% /run/user

可以看到 `/dev/vda1` 是那个 `40G` 磁盘的分区，使用率是 100%（四舍五入），磁盘空间的确是快用完了，还剩 `354` MB 。

按照常理，我们搜索一下相似的磁盘设备名：

    # ls /dev/vd*

    /dev/vda  /dev/vda1  /dev/vdb

可以看到一个叫 `/dev/vdb` 的设备，由此猜测是那个未挂载的 `200G` 数据盘。

后来，通过阿里云的 `Web管理系统` 里看到这两个磁盘的设备名分别是：

    /dev/xvda  --  40G 的系统盘
    /dev/xvdb  --  200G 的数据盘

但 `文件服务器` 的操作系统里的确找没有这两个设备，由于 `40G` 的系统盘在系统里叫 `/dev/vda`，由此可以断定 `200G` 的数据盘就是 `/dev/vdb` 。

### 2.1 分区和挂载 ###

#### 2.1.1 分区 ####

我们使用 `parted` 工具来分区，由于已经确定了 `200G` 磁盘的设备名称是 `/dev/vdb`，使用下面的命令进行分区：

    # parted /dev/vdb

    (parted)    mklabel gpt

    // 直接帮你计算分区对齐, 并且是分配100%的容量, 不会剩余的空间.
    (parted)    mkpart primary ext4 0% 100%

    // 查看一下分区的结果
    (parted)    print

    Model: Virtio Block Device (virtblk)
    Disk /dev/vdb: 215GB
    Sector size (logical/physical): 512B/512B
    Partition Table: gpt

    Number  Start   End    Size   File system  Name     Flags
     1      1049kB  215GB  215GB  ext4         primary

    // 检查一下是否对齐了? 如果显示 1 aligned，就说明对齐了
    // 其中的 1 是分区编号，对应着 vdb1，我们这里就分了一个区
    (parted)    align-check optimal 1
    1 aligned

    // 退出 parted
    (parted)    quit

使用下面的命令可以看到，跟原来比，多了一个分区叫 `/dev/vdb1`：

    # ls /dev/vd*

    /dev/vda  /dev/vda1  /dev/vdb /dev/vdb1

#### 2.1.2 格式化 ####

现在我们需要对 `/dev/vdb1` 分区进行格式化，并格式化为常用的 `ext4` 格式。

    # mkfs.ext4 /dev/vdb1

    mke2fs 1.42.9 (4-Feb-2014)
    Filesystem label=
    OS type: Linux
    Block size=4096 (log=2)
    Fragment size=4096 (log=2)
    Stride=0 blocks, Stripe width=0 blocks
    587202560 inodes, 9395240448 blocks
    469762022 blocks (5.00%) reserved for the super user
    First data block=0
    286720 block groups
    32768 blocks per group, 32768 fragments per group
    2048 inodes per group
    Superblock backups stored on blocks:
        32768, 98304, 163840, 229376, 294912, 819200, 884736, 1605632, 2654208,
        4096000, 7962624

    Allocating group tables: done
    Writing inode tables: done
    Creating journal (32768 blocks): done
    Writing superblocks and filesystem accounting information: done

格式化 `200G` 左右的磁盘，一般不到一秒即可完成，上面的输出信息仅供参考，不是实际的输出信息。

注意：前面的两个命令 “`parted /dev/vdb`” 和 “`mkfs.ext4 /dev/vdb1`” 中的 “`/dev/vdb`” 和 “`/dev/vdb1`” 一定不能敲错，回车之前检查一遍，如果写成了 `/dev/vda` 和 `/dev/vda1`，系统盘的数据瞬间就灰飞烟灭了，切记，我们要分区和格式化的对象是 `200G` 的数据盘！

#### 2.1.3 挂载分区 ####

我们准备把 `/dev/vdb1` 分区挂载到 `/data` 或者 `/mirror` 目录上，这个名字可以随意取，跟别的软件不冲突即可。

先创建这个 `/data` 目录：

    # mkdir -p /data

再编辑系统的 `/etc/fstab` 配置文件：

    # vim /etc/fstab

    # /etc/fstab: static file system information.
    #
    # Use 'blkid' to print the universally unique identifier for a
    # device; this may be used with UUID= as a more robust way to name devices
    # that works even if disks are added and removed. See fstab(5).
    #
    # <file system> <mount point>   <type>  <options>       <dump>  <pass>
    UUID=af414ad8-9936-46cd-b074-528854656fcd / ext4 errors=remount-ro 0 1

这里可以看到，阿里云是使用 `UUID` 作为标识来挂载 `/dev/vda1` 分区的，但我们在阿里云的 `Web` 管理系统里怎么也找不到 `云盘` 所对应的 `UUID` 值在哪里定义的，所以这里就跟平常物理机里的一样，使用设备名 `/dev/vdb1` 来挂载，在这个文件的末尾添加下面这一行：

    /dev/vdb1       /data            ext4    defaults          0       1

它的作用是当系统启动的时候，把分区 `/dev/vdb1` 挂载到 `/data` 目录上，保存后需要重启系统才能生效。

当然，我们也可以手动挂载（立即生效，无需重启），命令如下：

    # mount /dev/vdb1 /data

但是使用这个命令挂载的话，重启以后该 `mount` 命令是会失效的，必须要把挂载配置写到 `/etc/fstab` 文件里才能保证每次启动都自动挂载。

**是否挂载成功了?**

要检查是否挂载成功了，用 `cd` 命令进入 `/data` 目录看一下就知道了，也可以这样查看：

    # df -h

    Filesystem      Size  Used Avail Use% Mounted on
    udev            7.9G  4.0K  7.9G   1% /dev
    tmpfs           1.6G  420K  1.6G   1% /run
    /dev/vda1        40G  8.6G   29G  24% /
    none            4.0K     0  4.0K   0% /sys/fs/cgroup
    none            5.0M     0  5.0M   0% /run/lock
    none            7.9G     0  7.9G   0% /run/shm
    none            100M     0  100M   0% /run/user
    /dev/vdb1       197G   59G  129G  32% /data

最后一行，就表明了 `/dev/vdb1` 已成功挂载到了 `/data` 目录。

### 2.2 拷贝文件 ###

为了方便以后扩展，我们拷贝的时候，把原来的文件结构也照搬过来，这样比较灵活，也比较清晰一点。

（1） 在 `/data` 目录下面新建 `/data/var/lib/webfiles/` 子目录，一个命令就可以搞定:

    # mkdir -p /data/var/lib/webfiles/

（2） 把 `/var/lib/webfiles/` 目录下面的所有文件和文件夹复制到 `/data/var/lib/webfiles` 目录:

    # cp -p -r /var/lib/webfiles/. /data/var/lib/webfiles/

注：`-p` 参数表示 `same as --preserve=mode,ownership,timestamps`，即保留源文件或文件夹的模式，归属关系，时间戳等。

（3） 复制完成以后，先把原来的 `webfiles` 文件夹改名为 `webfiles_old`：

    # mv /var/lib/webfiles /var/lib/webfiles_old

这样做的好处是，如果复制或者软链接失败，原来的文件和文件夹还是存在的，以免丢失，等确定都成功了，再把原来的目录删掉。

（4） 把新建的目录 `/data/var/lib/webfiles` 软链接原来的 `/var/lib/webfiles` 目录上，命令如下：

    # ln -s /data/var/lib/webfiles /var/lib/webfiles

（5） 查看软连接是否成功，命令如下：

    # ll /var/lib/web*

    lrwxrwxrwx 1 root root        23 May 26 17:56 webfiles -> /data/var/lib/webfiles/
    -rw-r--r-- 1 root root 881408246 Sep 23  2016 webfiles.tar.gz

如果看到 `webfiles -> /data/var/lib/webfiles/` 就说明软链接成功了，使用 `cd /var/lib/webfiles` 命令进去看看文件是否正确。

### 2.3 删除原目录 ###

确定软链接和文件都正确以后，使用 `3. Web 验证` 小节的方式验证文件是否可以正常访问，再执行删除 `/var/lib/webfiles_old` 目录的命令，命令如下：

    # rm -r -f /var/lib/webfiles_old

如果发现有文件被其他进程占用，删除命令无法正常返回，可以在 `SSH` 终端里按下 `Ctrl + C` 来中断卡住的删除命令，重启系统，再执行一遍该命令即可。

## 3. Web 验证 ##

图片验证的地址：

[http://www.example.com/AdvertisementImage/2016-12-31/33f2ad80b42d40da9cc2d9a9c82682a3.jpg](http://www.example.com/AdvertisementImage/2016-12-31/33f2ad80b42d40da9cc2d9a9c82682a3.jpg)

## 4. 其他有用的命令 ##

（1） 查看一个目录下所有文件和子文件夹的总大小，比如 `/home` 目录:

    $ sudo du -bsh /home

（2） 查询一个 `/var` 目录下前 `10` 个最大的文件或文件夹：

    // (文件大小以KB为单位)
    $ sudo du -a /var | sort -n -r | head -n 10

    // (文件大小以MB为单位)
    $ sudo du -am /var | sort -n -r | head -n 10

<.End.>
