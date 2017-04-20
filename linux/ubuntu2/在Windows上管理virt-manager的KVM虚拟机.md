在 Windows 上管理 virt-manager 的 KVM 虚拟机
===============================================

（注：本文是基于 `Ubuntu 14.04 - 64 bit` 撰写的。）

# 1. 目的 #

我们想在 `Windows` 上直接通过某个 `APP` 来管理 `Linux` 上的 `KVM` 虚拟机。虽然可以在带界面的 `Linux` 上使用 `virt-manager` 来管理 `KVM` 虚拟机，但是我们甚至不想安装这个带界面的 `Linux`，直接利用现有的无界面的 `Linux` 服务器来实现我们的目的。

# 2. 准备工作 #

需要准备几个工具，在 `Windows` 上需要一个叫 `XManager` 的软件，它可以让我们通过 `xterm` 以 `X11Forwarding` 的方式把界面转发到 `Windows` 上。可以百度或Google搜索 “`XManager 5`”，下载原版的（不要下载那些破解版的），注册码是：`101210-450789-147200` 。

而在 `Linux`（`Ubuntu 14.04`）服务器上，我们需要 `xterm`，`virt-manager` 和 `ssh-askpass-gnome` 等几个组件。当然，`ssh-server`，`libvirt-bin` 这几个工具假设是早就安装好了的，因为没有它们，我们无法用 `SSH` 登录服务器和玩转 `KVM` 。

# 3. 怎么安装 #

## 3.1 Windows 客户端 ##

关于 `XManager` 的安装，就不多介绍了。`XManager` 是由 `Xstart`，`Xmanager - Passive`，`Xmanager - Broadcast`，`Xconfig`，`Xbrowser` 几个部分组成，我们只用到了 `Xstart` 和 `Xmanager - Passive` 两个应用。

## 3.2 Linux 服务器端 ##

随便挑选一台服务器来安装 `virt-manager`，我们是在 `Windows` 上利用 `Xstart` 连接该服务器的 `virt-manager`，来管理本地或其他服务器的 `KVM` 虚拟机的。

但是，有一个前提，就是必需先在该服务器上安装好 `libvirt` 的相关组件，因为 `virt-manager` 是基于 `libvirt` 工作的。

1）安装 `libvirt` 组件

如果这台服务器没有装过 `libvirt`，那么，需要先安装 `libvirt` 相关组件：

```shell
$ sudo apt-get install -y qemu-kvm qemu-system qemu-utils libvirt-bin pm-utils virtinst
```

如果之前你的系统已经在运行 `libvirt` 了，那么这一步是可以跳过的，并且上面这个安装未包括 `bridge-utils` 等组件，如果你想让这台服务器本身也提供 `libvirt` 的虚拟机服务，还需要做些别的配置，具体可参阅其他相关文章。

由于我们并不一定要在这个台服务器上提供 `KVM` 虚拟机服务，所以 `bridge-utils` 等组件是可以不用安装的。

2）安装 `virt-manager` 等组件

由于我们的服务器是没有 `GUI` 界面的，所以一般是没有必要安装 `virt-manager` 的。而现在安装它，是为了 `Windows` 客户端而安装的，并且我们的服务器并不需要真的安装 `GUI` 桌面支持，就可以在 `Windows` 端运行 `Linux` `GUI` 界面。

此外，我们还需要一个叫 `xterm` 的组件，通过它使用 `X11Forwarding` 的方式来把 `GUI` 界面转发到 `Windows` 客户端上。

还有一个小问题，由于 `virt-manager` 在新建一个服务器连接时，第一次登录，会弹出一个窗口，要你输入 `yes/no` 来确认，这个窗口需要一个组件来实现，所以还需要一个叫 `ssh-askpass-gnome` 的组件。

注：其实 `virt-manager` 的错误提示里给出的组件名称并不是这个，而是 `openssh-askpass`，而 `Ubuntu 14.04` 并没有这个组件，我们用 “`apt-cache search askpass`” 命令看到 `ssh-askpass-gnome` 这个组件，就试了一下，果然是这个。后来想了一下，`askpass` 的也很形象，“`询问密码`” 。

所以在 `Linux` 服务端，我们总共需要安装下列几个组件：

```shell
$ sudo apt-get install -y xterm virt-manager ssh-askpass-gnome
```

# 4. 怎么配置 #

## 4.1 关于配置 SSH ##

编辑 `/etc/ssh/sshd_config` 文件，注意不是 `/etc/ssh/ssh_config`，并且下面的设置放在后者里是会报错的。

编辑前者并添加下列内容：

```shell
$ sudo vim /etc/ssh/sshd_config

# What ports, IPs and protocols we listen for
Port 22

AllowUsers skyinno
PasswordAuthentication yes

X11Forwarding yes
X11UseLocalhost no
#X11DisplayOffset 10
```
* `Port 22`：是 `SSH` 服务的端口；
* `AllowUsers skyinno`： 表示运行登录的用户是 `skyinno`，这个用户最好不要设为 `root` 用户，有的系统会有 `warning`；
* `PasswordAuthentication yes`： 表示需要验证密码；
* `X11Forwarding yes`： 表示使用 X11 转发，这是设置 `ssh` 的主要目的；
* `X11UseLocalhost no`： 表示不允许转发本机 `localhost` 的请求，防止循环。

然后执行下列命令，重启 `ssh` 服务：

```shell
$ sudo /etc/init.d/ssh restart
或者
$ sudo service ssh restart
```

## 4.2 关于配置 Xstart ##

启动 `XManager` 的 `Xstart`，设置如下所示：

![XManager Xstart Config](./images/virt-manager/xmanager_xstart_config.png)

“`命令(C)`” 那一栏，是可以点文本框左下角的那个带左箭头的小图标，然后选菜单里面的 `xterm (Linux)` 直接填入的，当然，也可以自己手工输入 `/usr/bin/xterm -ls -display $DISPLAY` 。

# 5. 怎么使用 #

## 5.1 xxx ## 




