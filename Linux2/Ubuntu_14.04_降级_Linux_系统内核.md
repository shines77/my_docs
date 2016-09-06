
Ubuntu 14.04 降级 Linux 系统内核
-----------------------------------

升级 `linux` 内核，普遍使用系统提供的功能：

    $ sudo apt-get dist-upgrade

降级就比较麻烦了，需要我们自己手动安装。

下面以 `Ubuntu 14.04` 为例，从内核版本 `linux-headers-3.13.0-44` 降级到 `linux-headers-3.13.0-24` 为例，来演示下怎么降级 `linux` 内核。

    $ sudo aptitude install -y linux-image-3.13.0-24-generic linux-headers-3.13.0-24

然后：

    $ grep submenu /boot/grub/grub.cfg

看到父选项：

    Advanced options for Ubuntu

执行命令：

    $ grep menuentry /boot/grub/grub.cfg

看到子选项：

    Ubuntu, with Linux 3.13.0-24-generic

其他版本也可以从执行结果中选取相应的。

然后修改 `/etc/default/grub` 文件：

    $ sudo vim /etc/default/grub

将其中的：

    GRUB_DEFAULT=0

改为：

    GRUB_DEFAULT="Advanced options for Ubuntu>Ubuntu, with Linux 3.13.0-24-generic"

然后执行：

    $ sudo update-grub

此时如果有错误，则检查 `GRUB_DEFAULT` 是否需要将前面部分 ”Advanced options for Ubuntu>” 去掉后重新执行，可能你的不在次级选项中。

没有什么错误提示，即可 `reboot` 重启。

重新连接上后，检查 `uname -a`，如果已经到目的版本，则说明成功了。

## 参考 ##

降级linux内核，以Ubuntu 14.04为例，降级系统内核<br/>
[http://www.oldcai.com/archives/1026](http://www.oldcai.com/archives/1026)
