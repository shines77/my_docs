在 Windows 上管理 virt-manager 的 KVM 虚拟机
===============================================

# 1. 目的 #

我们想在 `Windows` 上直接通过某个 `APP` 来管理 `Linux` 上的 `KVM` 虚拟机。

虽然可以在带界面的 `Linux` 上使用 `virt-manager` 来管理 `KVM` 虚拟机，但是我们不想安装这个带界面的 `Linux`，直接利用现有的无界面的 `Linux` 服务器来实现我们的想法。

# 2. 准备工作 #

我们需要准备几个工具，`Windows` 上需要一个叫 `XManager` 的软件，它可以让我们通过 `xterm` 以 `X11Forwarding` 的方式把界面转发到 `Windows` 上。可以百度或Google搜索 “`XManager 5`”，下载原版的（不要下载那些破解版的），注册码是：`101210-450789-147200` 。

而在 `Linux`（Ubuntu 14.04）服务器上，我们需要 `xterm`，`virt-manager` 和 `ssh-askpass-gnome` 等几个组件。当然，`ssh-server`，`libvirt-bin` 这几个工具应该早就安装好了，因为这是必备的，没有这些你怎么登录服务器和玩 `KVM` 嘛。

# 3. 怎么安装 #

## 3.1 Windows 客户端 ##

关于 `XManager` 的安装，就不多介绍了。`XManager` 是由 `Xstart`，`Xmanager - Passive`，`Xmanager - Broadcast`，`Xconfig`，`Xbrowser` 几个部分组成，我们只用 `Xstart` 或 `Xmanager - Passive` 两个应用。

## 3.2 Linux 服务器端 ##

1）前期准备工作

我们随便挑选一台服务器来安装 `virt-manager`，但是使用 `virt-manager` 的前提是必需安装 `libvirt` 等相关组件。

如果你没有装过 `libvirt`，那么，需要先安装 `libvirt` 相关组件：

```shell
$ apt-get install -y qemu-kvm qemu-system qemu-utils libvirt-bin pm-utils virtinst
```

如果之前你的系统已经在跑 `libvirt` 了，那么这部是可以跳过的，但是上面这个安装还未包括 `bridge-utils` 等组件，由于我们并不一定要在这个台服务器上提供 `KVM` 虚拟机服务，所以是可以不必安装桥接组件的，当然，你如果想让这台服务器本身也提供 `libvirt` 的虚拟机服务，还需要做些别的配置，具体可参阅更详细的文章。

2）组件安装

由于我们的服务器是没有 `GUI` 界面的，所以一般我们没有必要安装 `virt-manager`，因为我们没有 `GUI` 桌面环境，但现在是为了 `Windows` 客户端而安装的，并且我们不必在服务器真的安装 `GUI` 桌面支持，很nice。

由于我们还需要一个东西叫 `xterm`，用它通过 X11Forwarding 的方式来把 `GUI` 界面转发 `Windows` 客户端上。

同时，由于 `virt-manager` 在第一次连接一个服务器的时候，会有个弹出窗口，要你输入 `yes/no` 来确认，所以还需要一个叫 `ssh-askpass-gnome` 的组件。`virt-manager` 的错误提示里给出的组件名称并不是这个，而是 `openssh-askpass`，而 `Ubuntu 14.04` 并没有这个组件，我们用 “`apt-cache search askpass`” 命令看到 `ssh-askpass-gnome` 这个组件，就试了一下，果然是这个。后来想了一下，`askpass` 的也很形象，“`询问密码`” 。

所以在 `Linux` 服务端，我们需要安装下列组件：

```shell
$ apt-get install -y xterm virt-manager ssh-askpass-gnome
```

# 4. 怎么使用 #

## 4.1 




