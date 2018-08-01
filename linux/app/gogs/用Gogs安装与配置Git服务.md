
使用 Gogs 搭建自己的 Git 服务器
---------------------------------

----------
下载安装
----------

## 1. 环境要求 ##

- 数据库（选择以下一项）：
    - [MySQL](http://dev.mysql.com)：版本 >= 5.5.3
    - [PostgreSQL](http://www.postgresql.org/)
    - 或者 **什么都不安装** 直接使用 SQLite3 或 TiDB
- [git](http://git-scm.com/)（bash）：
    - 服务端和客户端均需版本 >= 1.7.1
    - Windows 系统建议使用最新版
- SSH 服务器：
    - **如果您只使用 HTTP/HTTPS 或者内置 SSH 服务器的话请忽略此项**
    - 推荐 Windows 系统使用 [Cygwin OpenSSH](http://docs.oracle.com/cd/E24628_01/install.121/e22624/preinstall_req_cygwin_ssh.htm) 或 [Copssh](https://www.itefix.net/copssh)

### 1.1 安装数据库 ###

Gogs 支持 MySQL、PostgreSQL、SQLite3 和 TiDB（实验性支持），请根据您的选择进行安装：

- [MySQL](http://dev.mysql.com/downloads/mysql/)（引擎：INNODB）
- [PostgreSQL](http://www.postgresql.org/download/)

**注意事项:** 您可以使用 `/etc/mysql.sql` 来自动创建名为 `gogs` 的数据库。如果您选择手动创建，请务必将编码设置为 `utf8mb4`。

#### mysql 5.x (On Ubuntu 14.04) ####

	$ sudo apt-get install mysql-common-5.6 mysql-client-5.6 mysql-server-5.6 mysql-proxy

### 1.2 安装其它要求 ###

#### Mac OS X ####

假设您已经安装 [Homebrew](http://brew.sh/)：

	$ brew update
	$ brew install git

#### Debian/Ubuntu ####

	$ sudo apt-get update
	$ sudo apt-get install git

#### Windows ####

[下载并安装 Git](http://git-scm.com/downloads)

## 2. 安装 Gogs ##

### 2.1 新建用户 ###

`Gogs` 默认以 `git` 用户运行（你应该也不会想一个能修改 `ssh` 配置的程序以 `root` 用户运行吧？），运行：

	$ sudo adduser --system --home /home/git --shell /bin/bash --group git

新建好 `git` 用户，创建 `git` 用户，同时创建 `git` 用户组 。

如果创建 `git` 用户的时候没有设置密码，则可以使用 `passwd` 设置密码，如下：

	$ sudo passwd git

	# （然后输入两次密码）

同时，把 `git` 用户添加到 `sudoers` 组，切换到 `root` 用户，使用 `visudo` 命令：

	# visudo -f /etc/sudoers

找到 `root ALL=(ALL) ALL` 这么一行，添加下面内容：

	git ALL=(ALL) ALL

因为`/etc/sudoers` 文件是只读的，即使是在 `root` 用户下，编辑以后也是不能直接保存的，除非先去掉该文件的只读属性再编辑，所以还是使用 `visudo` 命令比较方便一点。

登录到 `git` 用户，并在 `git` 用户的主目录中新建好 `.ssh` 文件夹。

	$ su git
	$ cd /home/git
	$ mkdir .ssh

### 2.2 下载安装 Gogs ###

下面是两种安装方式的官方文档：

- [二进制安装](http://gogs.io/docs/installation/install_from_binary.html)
- [源码安装](http://gogs.io/docs/installation/install_from_source.html)

#### 2.2.1 下载解压

我使用的是二进制安装，需要从源码编译的话，请参考一般 `Go` 语言项目的编译。你可以把 `gogs` 装到你喜欢的目录下，例如 `/usr/share/gogs/` 或者 `/home/git/gogs/` ，然后下载并解压。

	$ cd /home/git
	$ wget -c https://dl.gogs.io/gogs_v0.9.13_linux_amd64.tar.gz
	$ mkdir gogs
	$ tar -zxf gogs_v0.9.13_linux_amd64.tar.gz -C ./

然后 gogs 目录下面是这样的：

	$ cd gogs
	$ ls -alh

	total 31M
	drwxr-xr-x  5 git git 4.0K Mar 20 03:19 ./
	drwxr-xr-x  4 git git 4.0K Jun 23 21:59 ../
	-rwxr-xr-x  1 git git  31M Mar 20 03:19 gogs*
	-rw-r--r--  1 git git 1.1K Mar 20 03:19 LICENSE
	drwxr-xr-x  7 git git 4.0K Mar 20 03:19 public/
	-rw-r--r--  1 git git 7.2K Mar 20 03:19 README.md
	-rw-r--r--  1 git git 5.1K Mar 20 03:19 README_ZH.md
	drwxr-xr-x  7 git git 4.0K Mar 20 03:19 scripts/
	drwxr-xr-x 10 git git 4.0K Mar 20 03:19 templates/

#### 2.2.2 开始安装 ###

首先建立好数据库，在 `Gogs` 目录的 `scripts/mysql.sql` 文件是数据库初始化文件。执行：

	$ mysql -u root -p < scripts/mysql.sql		#（需要输入密码）

即可初始化好数据库。

然后登录 `MySQL` 创建一个新用户 `gogs`，并将数据库 `gogs` 的所有权限都赋予该用户：

	$ mysql -u root -p

	> # （输入密码）
	> create user 'gogs'@'localhost' identified by '{你的mysql密码}';
	> grant all privileges on gogs.* to 'gogs'@'localhost';
	> flush privileges;
	> exit;

把 `Gogs` 运行起来：

	$ ./gogs web &

然后访问 `http://你的服务器IP:3000/` 来进行安装，填写好表单之后提交就可以了。

需要注意的是，`0.6.9.0903 Beta` 版本有个 `bug`，允许在关闭注册的情况下不添加管理员，这样安装完成之后将没有任何用户可以登录，所以请务必在安装界面指定一个管理员帐号。

如果想在系统启动的同时就在后台启动 gogs，则可以把 gogs 安装为服务：

    $ su git
    $ sudo cp ~/gogs/scripts/init/debian/gogs /etc/init.d/gogs
    $ sudo chmod 755 /etc/init.d/gogs
    $ sudo /etc/init.d/gogs start
    $ sudo update-rc.d gogs defaults    # 设置为开机启动
    $ sudo update-rc.d -f gogs remove   # 设置为开机时不启动

See: http://fullrec.github.io/2014/10/18/ubuntu-install-gogs-setup-git-server/

注意：下面修改 `/etc/rc.local` 和 `/etc/profile` 脚本设置为后台自启动 `gogs` 的方法是行不通的，也不推荐，写出来仅供参考。

 **以下内容仅供参考**

如果想在系统启动的同时就在后台启动 `gogs`，则可以这么做：

	$ sudo vim /etc/rc.local

添加下列内容：

    if [ $(id -u) -eq 0 ]; then
        su -c "/home/git/gogs/gogs web &" git
    fi

因为我们设置了只能使用 `git` 用户才能启动 `gogs`，所以使用命令 `su -c "要执行的命令" 指定的用户名` 来以 `git` 用户执行 `gogs` 的启动命令。（`-c` 是 `--command` 的意思）
（注：`su -c "/home/git/gogs/gogs web &" git` 命令目前只在 `Ubuntu 14.04` 上验证有效，别的版本或系统可能命令顺序不一定是这样的。）

同时, 把这些命令也加到 `/etc/profile` 文件里：

    $ sudo vim /etc/profile

    if [ $(id -u) -eq 0 ]; then
		sudo -u git "/home/git/gogs/gogs web &"
		# su -c "/home/git/gogs/gogs web &" git
    fi

下面这个代码是不OK的，不过也是一种思路，仅供研究。这个方法是先切换到 git 用户，再启动 gogs，但又不能退出 git 用户，因为用 exit 退出 git 用户，以 git 用户启动的 gogs 进程也就退出了，然而如果使用 su root 返回 root 用户则是需要输入管理员密码的，在启动脚本里这样做是不合理的，除非使用免输入密码的方法（可以，但比较麻烦）。所以这个方法不太可行。

	if [ "`id -u`" -eq 0 ]; then
		cur_dir=`pwd`
		su git
		cd /home/git/gogs
		sudo ./gogs web &
        su root
		cd $cur_dir
	fi

## 3. 配置与运行 ##

### 3.1 配置文件 ###

#### 3.1.1 默认配置文件 ####

默认配置都保存在 `conf/app.ini`，您 **永远不需要** 编辑它。该文件从 `v0.6.0` 版本开始被嵌入到二进制中。

#### 3.1.2 自定义配置文件 ####

那么，在不允许修改默认配置文件 `conf/app.ini` 的情况下，怎么才能自定义配置呢？很简单，只要创建 `custom/conf/app.ini` 就可以！在 `custom/conf/app.ini` 文件中修改相应选项的值即可。

例如，需要改变仓库根目录的路径：

	[repository]
	ROOT = /home/jiahuachen/gogs-repositories

当然，您也可以修改数据库配置：

	[database]
	PASSWD = root

#### 为什么要这么做？ ####

乍一看，这么做有些复杂，但是这么做可以有效地保护您的自定义配置不被破坏：

- 从二进制安装的用户，可以直接替换二进制及其它文件而不至于重新编写自定义配置。
- 从源码安装的用户，可以避免由于版本管理系统导致的文件修改冲突。

#### 3.1.3 运行 Gogs 服务 ####

#### 1) 开发者模式 ####

- 您需要在 `custom/conf/app.ini` 文件中将选项 `security -> INSTALL_LOCK` 的值设置为 `true`。
- 您可以使用超能的 `make` 命令：

	`
	$ make
	`<br/>
	`
	$ ./gogs web
	`

- 您可以在 Gogs 源码目录使用命令 `bra run`：
	- 安装 [bra](https://github.com/Unknwon/bra) 工具：<br/>
	`go get -u github.com/Unknwon/bra`

#### 2) 部署模式 ####

**脚本均放置在 `scripts` 目录，但请在仓库根目录执行它们**

- Gogs 支持多种方式的启动：
	- 普通：只需执行 `./gogs web`
	- 守护进程：详见 [scripts](https://github.com/gogits/gogs/tree/master/scripts) 文件夹
- 然后访问 `/install` 来完成首次运行的配置工作

#### 3.1.4 配置调整 ####

配置文件位于 `Gogs` 目录的 `custom/conf/app.ini`，是 `INI` 格式的文本文件。详细的配置解释和默认值请参考官方文档，其中关键的配置大概是下面这些：

- RUN_USER 默认是 git，指定 Gogs 以哪个用户运行
- ROOT 所有仓库的存储根路径
- PROTOCOL 如果你使用 nginx 反代的话请使用 http，如果直接裸跑对外服务的话随意
- DOMAIN 域名。会影响 SSH clone 地址
- ROOT_URL 完整的根路径，会影响访问时页面上链接的指向，以及 HTTP clone 的地址
- HTTP_ADDR 监听地址，使用 nginx 的话建议 127.0.0.1，否则 0.0.0.0 也可以
- HTTP_PORT 监听端口，默认 3000
- INSTALL_LOCK 锁定安装页面
- Mailer 相关的选项

## 4. 参考文章 ##

[https://mynook.info/blog/post/host-your-own-git-server-using-gogs](https://mynook.info/blog/post/host-your-own-git-server-using-gogs)

.
