Ubuntu 14.04 SVN Server 的安装和使用
============================================

# 1. 安装 SVN #

## 1.1. SVN 的安装 ##

    $ sudo apt-get install subversion

## 1.2. SVN 根目录 ##

我们把 `/home/skyinno/svn/` 作为 `SVN` 仓库的根目录。

    $ mkdir -p /home/skyinno/svn

### 1.2.1. 为 SVN 根目录软连接 ###

（注：正常情况下是不需要做这一步的软链接的，这是笔者的特殊情况，请跳过此步。）

由于服务器的 `系统盘` 容量比较小，不希望 `SVN` 仓库的目录占用 `系统盘` 的空间，所以我们把 `/home/skyinno/svn` 软连接到 `/data/svn` 目录。（这里 `/data` 目录所在的磁盘容量比较大，共有 `35TB`。）

先在 `/data` 目录下创建 `svn` 目录：

    $ sudo mkdir -p /data/svn

然后设置软连接，把 `/home/skyinno/svn` 软连接到 `/data/svn` 目录：

    $ sudo ln -s /data/svn /home/skyinno/svn

当然，这里要求 `/home/skyinno/svn` 不能存在，如果 `/home/skyinno/svn` 文件夹已经存在，上面的命令会报错。如果该文件夹里面是空的，什么都没有，则可以使用 `rm` 命令直接删除该目录；如果不是空的，先把原来的 `/home/skyinno/svn` 目录改名为 `/home/skyinno/svn.old`，再把 `/home/skyinno/svn.old` 目录里的内容复制到 `/data/svn/` 目录下面，再做软连接。具体步骤如下：

    $ sudo mv /home/skyinno/svn /home/skyinno/svn.old
    $ sudo cp -r /home/skyinno/svn.old/. /data/svn/
    $ sudo ln -s /data/svn /home/skyinno/svn

# 2. 配置和使用 SVN #

## 2.1. 创建 SVN 仓库 ##

创建一个名为 `myrepo` 的仓库：

    $ sudo svnadmin create /home/skyinno/svn/myrepo

然后查看一下 `myrepo` 目录里的文件：

    $ cd /home/skyinno/svn/myrepo
    $ ll -h

    drwxr-xr-x 2 root root 4096 Oct 21 14:28 conf/
    drwxr-sr-x 6 root root 4096 Oct 21 14:29 db/
    -r--r--r-- 1 root root    2 Oct 21 14:28 format
    drwxr-xr-x 2 root root 4096 Oct 21 14:28 hooks/
    drwxr-xr-x 2 root root 4096 Oct 21 14:28 locks/
    -rw-r--r-- 1 root root  246 Oct 21 14:28 README.txt

进入 `/conf` 目录，这个目录下存放的都是这个仓库的配置文件：

    $ cd conf
    $ ll -h

    -rw-r--r-- 1 root root 1080 Oct 21 14:28 authz
    -rw-r--r-- 1 root root  885 Oct 21 14:28 hooks-env.tmpl
    -rw-r--r-- 1 root root  309 Oct 21 14:28 passwd
    -rw-r--r-- 1 root root 4002 Oct 21 14:28 svnserve.conf

其中 `svnserve.conf` 是 `SVN` 仓库的一些设置，`passwd` 是验证用户的用户名和密码，是明文的，未加密。`authz` 是验证和读写权限相关的设置。

## 2.2. 配置 SVN 仓库 ##

我们只需要修改 `svnserve.conf` 和 `passwd` 文件即可，`authz` 的配置稍微复杂一点，此处不详细介绍。

编辑 `svnserve.conf` 文件：

    $ cd /home/skyinno/svn/myrepo
    $ cd conf
    $ sudo vim svnserve.conf

找到配置文件里分别包含以下三行的语句：

    # anon-access = read
    # auth-access = write

    # password-db = passwd

把前面的注释 “`#` ” 去掉，记得顶格（最前面不能有空格），并修改为如下形式，`read` 改为 `none`：

    anon-access = none
    auth-access = write

    password-db = passwd

这里，我们不希望匿名用户浏览和访问仓库，所以 `anon-access` 设置为 `none`，一般默认设置为 `read` 。更详细的解释为：

    # 如果[general]前面有#号，则去掉#号（注释）
    [general]

    # 匿名访问的权限，可以是 read, write, none, 默认为 read。
    # 设置成 none 的意思就是不允许匿名用户访问（读/写）
    anon-access = none

    # 认证用户的权限，可以是 read, write, none, 默认为 write。
    auth-access = write

    # 密码数据库的路径，去掉前面的 “# ”
    password-db = passwd

修改 `passwd` 文件，配置用户名和密码：

    $ cd /home/skyinno/svn/myrepo
    $ cd conf
    $ sudo vim passwd

    [users]
    # harry = harryssecret
    # sally = sallyssecret
    shines77 = abcd5678
    xiaoji = abcd1234

最好按这个格式，前面顶格，`=` 号之间保留空格，以免出现不必要的错误。

## 2.3. 启动 SVN 服务 ##

启动服务的命令：

    $ sudo svnserve -d -r /home/skyinno/svn

其中 `-d` 表示以 Daemon 方式启动，`-r` 后面是 `svn` 仓库的 `root` 目录。如果你的服务器不止一个 `IP`，还可以指定要绑定的 `IP` 地址，例如：

    $ sudo svnserve -d -r /home/skyinno/svn --listen-host 192.168.3.225

检查 `SVN` 服务是否已经启动了，可以使用命令：

    $ netstat -ntlp | grep 3690

    tcp     0     0 0.0.0.0:3690    0.0.0.0:*       LISTEN     16630/svnserve

关闭服务的命令：

    $ sudo killall -9 svnserve

如果想让 `SVN` 跟随系统开机自动启动，可以把 `svnserve` 的启动参数写到 `/etc/rc.local` 文件里，例如：

    $ sudo vim /etc/rc.local

    # 此处文件头若干内容省略 ......

    svnserve -d -r /home/skyinno/svn
    exit 0

## 2.4. 导入 SVN 仓库 ##

前面只是在 `SVN` 服务器创建了一个名为 `myrepo` 的仓库而已，现在来说说怎么从客户端导入文件到 `SVN` 服务器的仓库上。

为了测试方便，我们可以直接在服务器上建个目录，并创建一个叫 `ReadMe.txt` 的文件，然后来测试导入，例如：

    $ mkdir -p /home/skyinno/svn_repo/myrepo/
    $ cd /home/skyinno/svn_repo/myrepo/
    $ vim ReadMe.txt

在 `ReadMe.txt` 文件里随意输入几个字母即可，保存退出，然后执行下面的命令，把目录 `/home/skyinno/svn_repo/myrepo/` 导入到服务器上的 `myrepo` 仓库里：

    $ sudo svn import -m "first submit." /home/skyinno/svn_repo/myrepo/ file:///home/skyinno/svn/myrepo/

    Adding         /home/skyinno/svn_repo/myrepo/ReadMe.txt
    Committed revision 1.

导入命令的具体格式是：

    $ sudo svn import -m "本次提交的详细描述信息" {本地仓库的路径} {远端仓库的URL}

如果 `远端仓库的URL` 就在同一台服务器上，则可以使用 `file://` 前缀修饰。

## 2.5. 导出 SVN 仓库 ##

在远程的机器上安装 `SVN` 客户端，来测试 `SVN` 仓库的导出，例如：`Windows` 下安装 `TortoiseSVN`，下载地址是：[http://tortoisesvn.tigris.org/](http://tortoisesvn.tigris.org/)。

在需要导出的目录的空白处，点击鼠标右键，选出 `SVN 检出` (`SVN Checkout`)，然后输入 “`svn://192.168.3.225/myrepo/`”，回车确定，即可完成导出。

如果没有 `GUI` 环境，也可以通过命令行来执行导出，如下所示：

    $ svn checkout svn://192.168.3.225/myrepo --username=shines77

或者将 `checkout` 缩写为 `co`，即：

    $ svn co svn://192.168.3.225/myrepo --username=shines77

# 3. 使用 Apache 配置 SVN #

## 3.1. 安装组件 ##

先安装必要的组件：

    $ sudo apt-get install apache2 libapache2-svn

## 3.2. 修改配置 ##

然后修改 `/etc/apache2/mods-available/dav_svn.conf` 文件，在后面添加如下内容：

    $ sudo vim /etc/apache2/mods-available/dav_svn.conf

    <Location /svn/myrepo>

      DAV svn
      #SVNPath /home/skyinno/svn/myrepo
      SVNParentPath /home/skyinno/svn
      AuthType Basic
      AuthName "My Subversion Repository"
      AuthUserFile /etc/apache2/dav_svn.passwd

      #<LimitExcept GET PROPFIND OPTIONS REPORT>
        Require valid-user
      #</LimitExcept>

    </Location>

注：如果需要用户每次登录时都进行用户密码验证，请将 `<LimitExcept GET PROPFIND OPTIONS REPORT>` 与 `</LimitExcept>` 两行注释掉。

此外，`SVNPath` 和 `SVNParentPath` 你只能选择一个，两个同时设置会报错，推荐只设置 `SVNParentPath` 的值。

保存完上面的文件以后，你需要重启一下 `Apache2` 服务：

    $ sudo /etc/init.d/apache2 restart
    或者
    $ sudo service apache2 restart

## 3.3. 设置验证密码 ##

在前面的 `dav_svn.conf` 文件里，我们把验证用户的文件定义为 `/etc/apache2/dav_svn.passwd`，该文件包含了用户授权的详细信息（密码是加密的）。我们现在开始编辑它，命令格式如下：

    $ sudo htpasswd -c /etc/apache2/dav_svn.passwd {user_name}

其中，`-c` 代表是新建 `passwd` 文件（会删掉旧的文件），所以这个参数只是在添加第一个用户的使用，否则旧的授权信息就被删掉了。，例如：

    $ sudo htpasswd -c /etc/apache2/dav_svn.passwd shines77
    $ sudo htpasswd /etc/apache2/dav_svn.passwd xiaoji
    $ sudo htpasswd /etc/apache2/dav_svn.passwd xiaoming
    $ sudo htpasswd /etc/apache2/dav_svn.passwd xiaocai
    $ sudo htpasswd /etc/apache2/dav_svn.passwd xiaowu
    $ sudo htpasswd /etc/apache2/dav_svn.passwd xiaohan
    $ sudo htpasswd /etc/apache2/dav_svn.passwd xiaodong
    $ sudo htpasswd /etc/apache2/dav_svn.passwd xiaoguo
    $ sudo htpasswd /etc/apache2/dav_svn.passwd guest

可以看到，创建第二个和第二个以后的用户信息时，都没有添加 `-c` 参数。

然后你就可以通过浏览器访问如下地址：

    http://localhost/svn/myrepo

其中 `localhost` 改成你的 `SVN` 服务器的 `IP` 地址即可。

# 4. 参考文章 #

1. Ubuntu 下 SVN 安装和配置<br/>
[http://zhan.renren.com/itbegin?gid=3602888498033631485&checked=true](http://zhan.renren.com/itbegin?gid=3602888498033631485&checked=true)

2. Ubuntu 下搭建 SVN 服务器（Subversion）<br/>
[https://my.oschina.net/huangsz/blog/176128](https://my.oschina.net/huangsz/blog/176128)

3. Ubuntu 安装和配置 SVN （里面有对如何配置 `authz` 文件更详细的说明）<br/>
[http://www.cnblogs.com/wuhou/archive/2008/09/30/1302471.html](http://www.cnblogs.com/wuhou/archive/2008/09/30/1302471.html)

4. SVN 命令使用详解<br/>
[http://blog.sina.com.cn/s/blog_963453200101eiuq.html](http://blog.sina.com.cn/s/blog_963453200101eiuq.html)

.
