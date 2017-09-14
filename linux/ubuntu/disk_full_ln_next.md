
解决 Linux 服务器 nfs 挂载目录磁盘已满的问题 (后续)
====================================================

# 1. 起因 #

阿里云的 `ECS` 服务器，数据盘是用 `NFS` 远程挂载实现服务器间的文件共享的，上次已经从 `40G` 的系统盘里分离出来了，并使用软链接的方式挂到了一个新的 `200G` 云盘上，但最近这个 `200G` 的云盘也快满了，所以需要扩容，目前的已经在阿里云购买了数据盘的扩容，当分区还没有扩容，这里就是讨论 “无损扩容” 的解决方案。

# 2. 解决步骤 #

## 2.1 查看当前磁盘使用情况 ##

首先，`SSH` 登录到服务器 ，查看当前磁盘分区和挂载的情况：

    $ df -h

    Filesystem      Size  Used Avail Use% Mounted on
    udev            7.9G  4.0K  7.9G   1% /dev
    tmpfs           1.6G   11M  1.6G   1% /run
    /dev/vda1        40G  9.9G   28G  27% /
    none            4.0K     0  4.0K   0% /sys/fs/cgroup
    none            5.0M     0  5.0M   0% /run/lock
    none            7.9G     0  7.9G   0% /run/shm
    none            100M     0  100M   0% /run/user
    /dev/vdb1       197G  149G   38G  80% /data

可以看到 `/dev/vdb1` 是那个 `200G` 数据盘，使用率是 80%（四舍五入），剩余磁盘空间不是很多了。

## 2.2 分区和挂载 ##

### 2.2.1 分区 ###

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

#### 2.2.2 格式化 ####

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

# X. 参考文章 #

1. [阿里云官网文档：ECS 利用快照创建磁盘实现无损扩容数据盘](https://help.aliyun.com/knowledge_detail/40591.html) from `aliyun.com`
    
    本文方案：对当前数据盘做快照，然后在 `ECS` 服务器的同一个地域购买一个临时云盘（同一个地域拷贝才快，理论上应该属于局域网行为），以此快照来建立云盘，据说可以节省数据来回拷贝的时间。然后把临时云盘挂载到 `ECS` 服务器上，然后对要扩容的数据盘（当前数据盘）进行重新分区和格式化，最后使用 `cp -r --preserve=all` 命令把文件从临时云盘上原样拷贝回重新分区和格式化后的数据盘上，达到扩容的目的，再 `umount` 临时云盘。

2. [阿里云官网文档：扩容数据盘_Linux](https://help.aliyun.com/document_detail/25452.html?spm=5176.7738060.2.2.qUkuci) from `aliyun.com`

    本文档是在原来的云盘上做无损扩容。这里有一点需要注意，如果磁盘原来是使用 `fdisk` 或 `parted` 分的区，则对磁盘重新分区也只能使用原来的分区命令来对磁盘重新分区，否则可能会丢失分区表和文件，该文分别介绍了如何使用 `fdisk` 和 `parted` 重新分区的方法。

3. [阿里云ECS服务器扩容数据盘挂载且不影响数据过程记录](http://www.laozuo.org/10910.html) from `laozuo.org`

    该文是无损扩容的比较详细的实例范例（使用 `fdisk` 重新分区）。

4. [阿里云磁盘扩容踩坑总结](http://www.mamicode.com/info-detail-1691928.html) from `mamicode.com`

    同上。

5. [阿里云ECS磁盘扩容步骤](http://guanglin.blog.51cto.com/3038587/1688807) from `51cto.com`

    同上，但本文还介绍了如何 “`对LVM逻辑卷扩容`” 。

<.End.>
