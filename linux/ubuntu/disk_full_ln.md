
解决 Linux 服务器 nfs 挂载目录磁盘已满的问题
====================================================

## 1. 需求 ##

我们有一台阿里云服务器，这里简称 `文件服务器`，项目中由于上传的文件把系统盘 `40G` 已经占完了，没空间了。原来这台机器上还装有一个`200G` 的磁盘，但没有挂载使用（硬件上已经启用了的），其实也是虚拟云盘，不过使用上跟物理磁盘基本没什么区别。现在的需求是把一些文件或文件夹迁移到那未使用的 `200G` 磁盘上，以解决系统盘被占满的问题。

系统架构是这样的设计的，共有 `Web1`, `Web2`, `Web3`, `Web4` 四台 `Web服务器` 通过 `nfs` 的方式，都把本地的 `/var/lib/mrmsfiles` 文件夹映射到了后端的这台文件服务器上的同名文件夹上，例如：

    # apt-get install nfs-common
    # mount 192.168.3.177:/var/lib/mrmsfiles /var/lib/mrmsfiles

这里假设 `192.168.3.177` 是那个文件服务器的 `IP地址` 。

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

可以看到 `/dev/vda1` 是那个 `40G` 磁盘的分区，使用率已经是 100%，磁盘的确是满了。

按照常理，我们搜索一下相似的磁盘设备：

    # ls /dev/vd*

    /dev/vda  /dev/vda1  /dev/vdb

可以看到一个叫 `/dev/vdb` 的设备，由此猜测是那个未挂载的 `200G` 磁盘。

后来通过阿里云的 Web 管理系统里看到这两个磁盘的设备名分别是：

    /dev/xvda  --  40G 的系统盘
    /dev/xvdb  --  200G 的数据盘

但 `文件服务器` 的操作系统里的确找到不到这两个设备，由于 `40G` 的系统盘在系统里叫 `/dev/vda`，由此可以断定 `200G` 的数据盘就是 `/dev/vdb` 。

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

现在我们需要对 `/dev/vdb1` 分区进行格式化，选择常用的 `ext4` 格式。

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

格式化一般磁盘小的话很快的，上面的输出信息仅供参考。

注意：这里和前面的 `parted` 命令里的 `设备名` 和 `分区` 千万要检查清楚再回车，不然把原来的系统盘的数据弄没了就 `GG` 了。

#### 2.1.3 挂载分区 ####

我们把 `/dev/vdb1` 分区挂载到 `/data` 或者 `/mirror` 目录，名字随意，跟别的软件不冲突即可。

先创建 `/data` 目录：

    # mkdir -p /data

编辑系统的 `fstab` 配置文件：

    # vim /etc/fstab

    # /etc/fstab: static file system information.
    #
    # Use 'blkid' to print the universally unique identifier for a
    # device; this may be used with UUID= as a more robust way to name devices
    # that works even if disks are added and removed. See fstab(5).
    #
    # <file system> <mount point>   <type>  <options>       <dump>  <pass>
    UUID=af414ad8-9936-46cd-b074-528854656fcd / ext4 errors=remount-ro 0 1

这里可以看到，阿里云是使用 `UUID` 作为标识来挂载 `/dev/vda1` 磁盘的分区的，但我在阿里云的 `Web` 管理系统里怎么也找不到 `云盘` 对应的 `UUID` 值是多少，所以这里就使用常用的设备名 `/dev/vdb1` 来挂载，在这个文件的后面添加下面这一行：

    /dev/vdb1       /data            ext4    defaults          0       1

作用是系统开机的时候，把分区 `/dev/vdb1` 挂载到 `/data` 目录，挂载需要重启系统才能生效。

当然，我们也可以手动挂载（立即生效，无需重启），命令如下：

    # mount /dev/vdb1 /data

但是使用这个命令挂载的话，重启以后是会失效的，所以必需要把挂载写到 `/etc/fstab` 配置文件里。

**是否挂载成功?**

要检查是否挂载成功了，进入 `/data` 目录一下就知道了，也可以这样：

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

最后一行就表明了 `/dev/vdb1` 成功挂载到了 `/data` 目录。

### 2.2 拷贝文件 ###

为了方便以后扩展，我们把原来的文件结构照搬过来，这样比较灵活，也比较清晰一点。

在 `/data` 文件夹下面创建 `/data/var/lib/mrmsfiles/` 目录，一个命令就可以搞定:

    # mkdir -p /data/var/lib/mrmsfiles/

拷贝 `/var/lib/mrmsfiles/` 下面的所以文件和文件夹拷贝到 `/data/var/lib/mrmsfiles` 文件夹下面:

    # cp -p -r /var/lib/mrmsfiles/. /data/var/lib/mrmsfiles/

注：`-p` 参数表示 `same as --preserve=mode,ownership,timestamps`，即保留源文件或文件夹的模式，归属关系，时间戳等。

复制完成以后，我们先把原来的 `mrmsfiles` 文件夹改名：

    # mv /var/lib/mrmsfiles /var/lib/mrmsfiles_old

这样做的好处是，如果复制或者软链接失败，原来的文件和文件夹还是存在的，以免丢失，等确定都成功了，再把原来的目录删掉。

下面把 `/data/var/lib/mrmsfiles` 目录软连接原来的 `/var/lib/mrmsfiles` 目录上，命令如下：

    # ln -s /data/var/lib/mrmsfiles /var/lib/mrmsfiles

查看软连接是否成功，命令如下：

    # ls /var/lib/mrms*

    lrwxrwxrwx 1 root root        23 May 26 17:56 mrmsfiles -> /data/var/lib/mrmsfiles/
    -rw-r--r-- 1 root root 881408246 Sep 23  2016 mrmsfiles.tar.gz

如果看到 `mrmsfiles -> /data/var/lib/mrmsfiles/` 就说明软链接成功了，使用 `cd /var/lib/mrmsfiles` 进去看看文件是否正确就知道了。

### 2.3 删除原目录 ###

确定软链接和文件都正确以后，使用第 3 小节的方式验证，再删除原来的文件夹 `/var/lib/mrmsfiles_old`，命令如下：

    # rm -r -f /var/lib/mrmsfiles_old

如果发现有文件被其他进程占用，无法完全删除，可以使用 `Ctrl+C` 暂停卡住的删除命令，重启系统，再执行一边该命令即可。

## 3. 验证 ##

厦门马拉松官网：

[http://www.xmim.org](http://www.xmim.org)

图片验证的地址：

[http://mrms.skyinno.com/AdvertisementImage/2016-12-31/33f2ad80b42d40da9cc2d9a9c82682a3.jpg](http://mrms.skyinno.com/AdvertisementImage/2016-12-31/33f2ad80b42d40da9cc2d9a9c82682a3.jpg)

## 4. 其他有用的命令 ##

查看一个目录下所有文件和子文件夹的总大小:

    $ sudo du -bsh /home

查询一个目录下前 `10` 个最大的文件或文件夹：

    // (文件大小以KB为单位)
    $ sudo du -a /var | sort -n -r | head -n 10

    // (文件大小以MB为单位)
    $ sudo du -am /var | sort -n -r | head -n 10

<.End.>