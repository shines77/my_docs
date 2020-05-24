# Ubuntu 16.04 搭建魔兽世界服务器(基于CMaNGOS WOW 1.12.1)

## 1. 安装相关环境

系统选择的是 `Ubuntu 16.04 (64bit)`，使用 `root` 用户登陆，先更新和安装相关的依赖库、编译工具以及 `mysql` 数据库。

```shell
sudo -i
apt update
apt upgrade
apt install -y build-essential gcc g++ automake autoconf patch make cmake git-core subversion libtool libssl-dev grep binutils zlibc libc6 libbz2-dev libboost-all-dev tmux screen libmysql++-dev mysql-server
```

注：安装过程中，会让你输入 `mysql` 的 `root` 用户密码，记得用记事本保存一下，以免忘记。

编译与安装需要的工具有：gcc, g++, git, make, cmake, zlib, boost, mysql-server, mysql-client 等。

运维所需要的工具：tmux, screen 等。

## 2. 新建用户

为了服务器的安全考虑，建议使用普通用户来做日常运维，一般不推荐直接使用 `root` 用户登陆。

先新建一个用来做日常运维的普通用户账号：

例如：用户名叫 `shines77`，所在的 `group` 是普通用户组 `users`。

```shell
useradd -m -d /home/shines77 -c "Operations staff" -g users shines77
```

注：如果你的 `ID` 叫 `Tom`，则把上面的 `shines77` 都改为 `tom` 即可，用户名建议都使用小写。

再新建一个叫 `mangos` 的用户，所在的 `group` 跟用户名一样都是 `mangos`，用于编译和运行 `cmongos`：

```shell
useradd -m -d /home/mangos -c "MaNGOS" -U mangos
```

创建了完以上两个用户之后，还要给这两个账号设置密码，否则无法登陆，使用以下命令：

```shell
passwd shines77
passwd mangos
```

每输入一条命令，都会要你输入该用户的密码，并再次确认该密码，同样的，记得用记事本把密码保存起来。

## 3. 编译和安装 CMaNGOS

上面我们新建了 `mangos` 用户，进入该用户的根目录：

```shell
cd /home/mangos
```

### 3.1 下载源码

从 `github` 上拉取 `CMaNGOS` 基于 `WoW 1.12.x` 版本的源代码：

```shell
git clone git://github.com/cmangos/mangos-classic.git mangos
```

拉取 `CMaNGOS` 基于 `WoW 1.12.x` 版本的数据库：

```shell
git clone git://github.com/cmangos/classic-db.git
```

此时，在 `/home/mangos` 目录下能看到多了 `mangos` 和 `classic-db` 两个目录。

### 3.2 编译和安装

#### 3.2.1 配置 makefile

在 `/home/mangos` 目录下新建一个 `build` 目录，并进入该目录：

```shell
mkdir /home/mangos/build
cd /home/mangos/build
```

接下来使用 `cmake` 来做编译配置，可以有下列三种选择：

* 只编译服务器程序，可以使用：

```shell
cmake ../mangos -DCMAKE_INSTALL_PREFIX=/opt/wowserver -DPCH=1 -DDEBUG=0
```

* 编译服务器程序和地图提取工具（推荐）：

```shell
cmake ../mangos -DCMAKE_INSTALL_PREFIX=/opt/wowserver -DBUILD_EXTRACTORS=ON -DPCH=1 -DDEBUG=0
```

* 编译服务器程序、地图提取工具，以及打开机器人：

```shell
cmake ../mangos -DCMAKE_INSTALL_PREFIX=/opt/wowserver -DBUILD_EXTRACTORS=ON -DPCH=1 -DDEBUG=0 -DBUILD_PLAYERBOT=ON
```

这里需要注意，`-DCMAKE_INSTALL_PREFIX=/opt/wowserver` 指的是 `cmangos` 的安装路径为 `/opt/wowserver`，所有的可执行程序和配置文件，都会拷贝到该文件夹中。

例如：`-DCMAKE_INSTALL_PREFIX=\../mangos/run` 表示安装到 `/home/mangos/mangos/run` 目录中，如果没有指定 `-DCMAKE_INSTALL_PREFIX` 参数，默认的安装路径是 `/opt/mangos` 。

更多的编译选项，请参考官方文档：[Installation Instructions](https://github.com/cmangos/issues/wiki/Installation-Instructions)

#### 3.2.2 编译 CMaNGOS

以上是用 `cmake` 配置 `CMaNGOS` 的编译设置，下面开始编译。

编译 `CMaNGOS` 和 `ScriptDev2`：

```shell
make
```

注：如果你的服务器是多核的 `CPU`，可以使用 `-j` 参数开启多线程编译，例如："`cmake -j8`" 表示采用 `8` 个线程编译，指定的线程数最好跟你服务器的 `CPU` 物理核心数一样，不包含 `CPU` 的超线程，比如 `4` 核 `8` 线程的 `CPU` 使用 `-j4` 即可，否则反而会降低效率。

然后是一个漫长的编译过程，请耐心等待。

#### 3.2.3 安装 CMaNGOS

编译完成后，我们开始安装：

```shell
make install
```

安装的路径已经在上一步中指定过了，即 `/opt/wowserver`。

安装完成后，需要复制配置文件并改名：

```shell
cd /opt/wowserver/etc
cp mangosd.conf.dist mangosd.conf
cp realmd.conf.dist realmd.conf
```

如果编译的时候选择了开启机器人，还需要复制：

```shell
cp playerbot.conf.dist playerbot.conf
```

#### 3.2.4 提取地图文件

提取地图文件可以在服务器端做，也可以自己在 `Windows` 下提取好了，或者在本地的 `Ubuntu` 虚拟机中提取好了，再上传到服务器上，都可以。为什么考虑先在本地提取地图文件，是因为整个提取过程是很耗时的，如果你的服务器不允许你这么做，或者性能太低，可以在本地提取完成后再上传。

另外，整个 `Data` 文件夹大约 `5.31` GB，上传到服务器上也是非常耗时的，提取出来的所有地图文件大约 `x.xx` GB，本地提取后再上传可以减少很多上传的文件大小。

下面分别介绍在服务器上提取和本地提取后再上传两种方式：

##### 3.2.4.1 在服务器端提取地图文件

1. 上传 `Data` 文件夹

上传 `WoW 客户端` 中的 `Data` 文件夹 (`\World of Warcraft\Data`) 到你的服务器下的 `cmangos` 安装目录下 `bin/tools` 文件夹下，这里可能需要 `WinSCP` 或 `SFTP` 服务器文件上传软件，请自行百度。

我们这里就是上传到 `/opt/wowserver/bin/tools` 目录下。

2. 给脚本添加运行权限

拷贝提取脚本，默认的提取地图脚本是没有运行权限的，我们给加上：

```shell
cd /opt/wowserver/bin/tools
chmod +x ExtractResources.sh
chmod +x MoveMapGen.sh
```

在最新的 `CMaNGOS` 版本中，如果编译时开启了提取地图，安装完成后，`/bin/tools/` 是直接包含提取地图的脚本的。

*注：* 如果你的安装路径下的 `/bin/tools/` 下没有提取地图的脚本文件，可以使用下面的命令复制到 `/bin/tools/` 目录下：

```shell
mv /home/mangos/mangos/contrib/extractor_scripts/* /opt/wowserver/bin/tools
```

然后再用上面的命令给提取脚本加上运行权限，最终 `/opt/wowserver/bin/tools` 目录中的文件列表如下所示：

```shell
root@Ubuntu64:/opt/wowserver/bin/tools# ll

total 10052
drwxr-xr-x 3 root root    4096 May 24 12:14 ./
drwxr-xr-x 3 root root    4096 May 24 09:38 ../
-rwxr-xr-x 1 root root  108008 May 24 09:30 ad*
drwxr-xr-x 3 root root    4096 May 24 18:14 Data/
-rwxr-xr-x 1 root root    6095 May 23 19:49 ExtractResources.sh*
-rwxr-xr-x 1 root root 5622136 May 24 09:30 MoveMapGen*
-rwxr-xr-x 1 root root    4470 May 23 19:49 MoveMapGen.sh*
-rw-r--r-- 1 root root     316 May 23 19:49 offmesh.txt
-rwxr-xr-x 1 root root 3655152 May 24 09:30 vmap_assembler*
-rwxr-xr-x 1 root root  865624 May 24 09:30 vmap_extractor*
```

3. 运行提取脚本

执行提取脚本，可能需要几个小时，看你服务器的配置：

```shell
./ExtractResources.sh
```

提取完成之后，会多出以下几个文件夹：`dbc`, `maps`, `mmaps`, `vmaps` 等，其他的如 `Buildings`, `Cameras` 等文件夹是无用的。

4. 移动地图文件

我们在 `CMaNGOS` 服务器的安装目录新建两个文件夹 `share` 和 `log`，分别用来存放地图文件和保存日志。

```shell
cd /opt/wowserver
mkdir share
mkdir log
```

并将相关的地图文件移动到 `/opt/wowserver/share` 目录中：

```shell
mv bin/tools/*maps share/
mv bin/tools/dbc share/
```

5. 删除 `Data` 文件夹

完成提取过程后，上传的 `Data` 文件夹就可以删掉了，这样能节省很多硬盘空间。

##### 3.2.4.2 在本地提取地图文件

xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

#### 3.2.5 安装数据库

先切换到 `/home/mangos` 目录：

```shell
cd /home/mangos
```

##### 3.2.5.1 创建空的数据库

```shell
mysql -uroot -p < mangos/sql/create/db_create_mysql.sql
```

##### 3.2.5.2 初始化数据库

初始化 `Mangos (world-db)` 数据库：

```shell
mysql -uroot -p classicmangos < mangos/sql/base/mangos.sql
```

初始化 `realmd` 数据库：

```shell
mysql -uroot -p classicrealmd < mangos/sql/base/realmd.sql
```

初始化 `characters` 数据库：

```shell
mysql -uroot -p classiccharacters < mangos/sql/base/characters.sql
```

##### 3.2.5.3 导入世界数据库

切换到 `classic-db` 目录：

```shell
cd /home/mangos/classic-db
```

运行一次 `InstallFullDB.sh` 文件，这会生成配置文件 `InstallFullDB.config`：

```shell
./InstallFullDB.sh
```

`InstallFullDB.config` 文件默认的内容如下：

```bash
####################################################################################################
# This is the config file for the './InstallFullDB.sh' script
#
# You need to insert
#   MANGOS_DBHOST:	Your MANGOS database host
#   MANGOS_DBNAME:	Your MANGOS database schema
#   MANGOS_DBUSER:	Your MANGOS username
#   MANGOS_DBPASS:	Your MANGOS password
#   CORE_PATH:    	Your path to core's directory
#   MYSQL:        	Your mysql command (usually mysql)
#
####################################################################################################

## Define the host on which the mangos database resides (typically localhost)
MANGOS_DBHOST="localhost"

## Define the database in which you want to add clean DB
MANGOS_DBNAME="classicmangos"

## Define your username
MANGOS_DBUSER="mangos"

## Define your password (It is suggested to restrict read access to this file!)
MANGOS_DBPASS="mangos"

## Define the path to your core's folder
##   If set the core updates located under sql/updates/mangos from this mangos-directory will be added automatically
CORE_PATH=""

## Define your mysql programm if this differs
MYSQL="mysql"

## Define if you want to wait a bit before applying the full database
FORCE_WAIT="YES"

## Define if the 'dev' directory for processing development SQL files needs to be used
##   Set the variable to "YES" to use the dev directory
DEV_UPDATES="NO"

# Enjoy using the tool
```

编辑 `InstallFullDB.config` 文件，并修改 `CORE_PATH` 和 `MANGOS_DBPASS` 参数。如下所示：

```bash
CORE_PATH="/home/mangos/mangos"
MANGOS_DBPASS="mangos"     ## 修改为你想要的数据库 mangos 用户的密码，不改也可以
```

再次运行 `InstallFullDB.sh` 文件，此时才正式开始导入世界数据库，并等待完成：

```shell
./InstallFullDB.sh
```

##### 3.2.5.4 导入其他数据库

其实下面的步骤已经在上一小节的 `InstallFullDB.sh` 脚本中执行过了，运行脚本后的结果中可以看到，没必要自己再执行一遍，这里只是把它们列出来，仅供参考。

初始化 `original_data` 相关数据库：

```shell
for sql_file in `ls mangos/sql/base/dbc/original_data/*.sql`; do mysql -uroot -p --database=classicmangos < $sql_file ; done
```

初始化 `cmangos_fixes` 相关数据库：

```shell
for sql_file in `ls mangos/sql/base/dbc/cmangos_fixes/*.sql`; do mysql -uroot -p --database=classicmangos < $sql_file ; done
```

```shell
mysql -uroot -p classicmangos < mangos/sql/scriptdev2/scriptdev2.sql
```

### 3.3 配置 CMaNGOS

#### 3.3.1 配置 mangos 服务器和认证服务器

关于配置服务器已经在 `3.2.3 安装 CMaNGOS` 小节里介绍了，这里再贴一遍，如果前面做过了，可以跳过此步。

复制配置文件并改名：

```shell
cd /opt/wowserver/etc
cp mangosd.conf.dist mangosd.conf
cp realmd.conf.dist realmd.conf
```

如果编译的时候选择了开启机器人，还需要复制：

```shell
cp playerbot.conf.dist playerbot.conf
```

#### 3.3.2 编辑 mangosd.conf

```shell
vim mangosd.conf
```

修改 `DataDir` 和 `LogsDir` 的值为：

```bash
DataDir = "../share"
LogsDir = "../log"
```

这两个目录是前面我们新建的，跟前面的目录名统一即可。`share` 是存放地图数据的目录，`log` 是存放日志的目录。

#### 3.3.3 编辑 realmd.conf

```shell
vim realmd.conf
```

修改 `LogsDir` 的值为：

```bash
LogsDir = "../log"
```

#### 3.3.4 更新数据库 realmd 的 realmlist

使用下面的命令登录 `mysql` 数据库，

```shell
mysql -uroot -p
```

在 `mysql` 的状态下输入：

```mysql
mysql> use classicrealmd;
```

先查看一下 `realmlist` 表的内容：

```sql
select * from realmlist;

+----+--------+-----------+------+------+------------+----------+------------+-------------+
| id | name   | address   | port | icon | realmflags | timezone | population | realmbuilds |
+----+--------+-----------+------+------+------------+----------+------------+-------------+
|  1 | MaNGOS | 127.0.0.1 | 8085 |    1 |          0 |        1 |          0 |             |
+----+--------+-----------+------+------+------------+----------+------------+-------------+
1 row in set (0.00 sec)
```

可以看到默认就已经添加了一个 `realm` (大区和服务器)。

注：由于显示的文字太长的问题，上面的查询结果中把 `allowedSecurityLevel` 的值移除了。

所以，我们有两种方式修改 `realmlist` 的设置：

1. 直接使用 Update 语句修改

`SQL` 语句如下：

```sql
UPDATE realmlist SET name='MaNGOS', address='127.0.0.1', port='8085', timezone='8' WHERE id=1;
```

2. 删除原来的数据，在插入一条新数据：

`SQL` 语句如下：

```sql
DELETE FROM realmlist WHERE id=1;
INSERT INTO realmlist (id, name, address, port, icon, realmflags, timezone, allowedSecurityLevel)
VALUES ('1', 'MaNGOS', '127.0.0.1', '8085', '1', '0', '8', '0');
```

**注意事项**

你必须保证修改的 `realmlist` 表中 `id`, `port` 的值与配置文件 `mangosd.conf` 中的"`RealmID`" 和 "`WorldServerPort`" 的值保持一致。

`timezone` 的设置为时区，中国处于东八区，即取值 `8` 即可。

这里，如果你想把服务器提供给外网的用户使用，而不是只在本地访问，需要把 `address` 的值 `127.0.0.1` 改为你的服务器的 `IP` 地址。

## 3.4 运行 CMaNGOS 服务器

建议大家使用 `screen` 或 `tmux` 来运行，不过也可以登陆多个 `SSH` 窗口执行，这里分别介绍 `screen` 和 `tmux` 两种方式。

### 3.4.1 使用 screen 运行

先简单介绍一下 `screen`，它是一个可以在多个进程（通常是交互式 `shell`）之间复用一个物理终端的全屏幕窗口管理器，即在 `Linux` 下使用多窗口。

常用的 `screen` 命令：

```shell
screen -ls（或者screen -list）    # 列出当前所有的 session
screen -S session_name           # 新建一个叫 session_name 的 session
screen -r session_name           # 回到 session_name 这个 session
screen -d session_name           # 远程 detach 某个 session (即暂时退出这个 session)
screen -d -r session_name        # 结束当前 session 并回到 session_name 这个 session
```

下面是处于 `screen` 状态下的一些常用组合键：

* `Ctrl` + `a` + `c`：创建一下新的 screen 窗口
* `Ctrl` + `a` + `p`：切换到上一个 screen 窗口
* `Ctrl` + `a` + `n`：切换到下一个 screen 窗口
* `Ctrl` + `a` + `d`：暂时退出当前 screen 窗口（等会还想连接这个 screen 窗口）
* `Ctrl` + `d` 或者输入 `exit`：退出当前 screen 窗口，结束当前 screen 窗口，不想再连接回来（即杀死会话）

注意：`Ctrl` + `a` + `c`，表示按着 `Ctrl` 键不放，分别按下 `a` 键和 `c` 键，以此类推。

1. 启动 `mangosd` 服务

（不开启机器人）：

```shell
screen -S mangosd
cd /opt/wowserver/bin
./mangosd /opt/wowserver/bin/mangosd -c /opt/wowserver/etc/mangosd.conf
```

（开启机器人）：

```shell
screen -S mangosd
cd /opt/wowserver/bin
./mangosd /opt/wowserver/bin/mangosd -c /opt/wowserver/etc/mangosd.conf -a /opt/wowserver/etc/playerbot.conf
```

`screen` 允许你在任何时候（只要不重启系统）恢复已经创建的命名窗口，所以执行上面的命令后，可以直接关掉 `SSH` 窗口，然后在任何使用想恢复该窗口时使用如下命令：

```shell
screen -R mangosd
```

即可返回启动 `mangosd` 服务的窗口。

2. 启动 `realmd` 服务

启动 `mangosd` 服务以后，可以再打开一个 `SSH` 窗口执行下面的命令：

```shell
screen -S realmd
cd /opt/wowserver/bin
./realmd /opt/wowserver/bin/realmd -c /opt/wowserver/etc/realmd.conf
```

同样的，如果想任何时候恢复 `realmd` 窗口可以使用命令：

```shell
screen -R realmd
```

这里不推荐使用 `Ctrl` + `a` + `c` 命令在同一个 `session` 里新建窗口，同时启动 `mangosd` 和 `realmd` 服务，因为这样不便于管理，我们分别为两个服务启动不同的 `session`，这样更清晰一点。

### 3.4.2 使用 tmux 运行

```shell
tmux
cd /opt/wowserver/bin
./mangosd -c ../etc/mangosd.conf                             // 不带机器人
./mangosd -c ../etc/mangosd.conf -a ../etc/playerbot.conf    // 带机器人
```

我们将 `tmux` 分开，先按下 `Ctrl + b` 然后按 `Shift + 5`，再按 `Ctrl + b` 接着按 `o` 键切换到另一半，输入：

```shell
cd /opt/wowserver/bin
./realmd -c ../etc/realmd.conf
```

## 3.5 开机自启动 CMaNGOS

上一节已经介绍了如果启动 `CMaNGOS` 服务器，但是这样是不够的，我们必须设置为开机自启动，这样才是一个专业的行为。

### 3.5.1 设置 mangos 的权限

将服务器安装路径 `/opt/wowserver` 的执行权限给 `mangos` 用户，并使用 `mangos` 用户来执行，非必须的，但推荐这样做。

```shell
chown -R mangos:mangos /opt/wowserver
su - mangos
```

### 3.5.2 自启动脚本

制作自启动脚本，如下：

```shell
cd /opt/wowserver/bin
touch cmangos.sh
vim cmangos.sh
```

然后写入如下内容：

（不开启机器人）

```bash
#!/bin/bash
exec screen -dmS mangosd /opt/wowserver/bin/mangosd -c /opt/wowserver/etc/mangosd.conf++
exec screen -dmS realmd /opt/wowserver/bin/realmd -c /opt/wowserver/etc/realmd.conf++
```

（开启机器人）

```bash
#!/bin/bash
exec screen -dmS mangosd /opt/wowserver/bin/mangosd -c /opt/wowserver/etc/mangosd.conf -a /opt/wowserver/etc/playerbot.conf++
exec screen -dmS realmd /opt/wowserver/bin/realmd -c /opt/wowserver/etc/realmd.conf++
```

给 `cmangos.sh` 脚本添加执行权限：

```shell
chmod +x cmangos.sh
```

### 3.5.3 开机启动

设置计划任务，这样就能开机自启动服务器程序了。

```shell
crontab -e
```

如果你是第一次运行 `crontab`，它会询问你使用哪个编辑器，可以选一个你更熟悉的，这里推荐选择 `3 - Vim (Basic)`。

然后在配置文件里添加这么一行：

```bash
@reboot /bin/bash /opt/wowserver/bin/cmangos.sh
```

### 3.6 开启相应的端口

开启 `3724` 和 `8085` 端口：

```shell
iptables -I INPUT -p tcp --dport 3724 -j ACCEPT
iptables -I INPUT -p tcp --dport 8085 -j ACCEPT

iptables -I ufw-user-input -p tcp -m tcp --dport 3724 -j ACCEPT
iptables -I ufw-user-input -p tcp -m tcp --dport 8085 -j ACCEPT
```

查看 `ipv4 iptables` 状态：

```shell
iptables -L -n -v
```

保存配置：

```shell
iptables-save > $HOME/firewall.txt       # 保存配置
iptables-restore < $HOME/firewall.txt    # 恢复配置

ufw reload                               # 重启 ufw 服务
ufw status                               # 查看 ufw 服务的端口是否已经开
```

查看服务器已在监听的端口的命令：

```shell
netstat -lntup
```

## 4. 参考文章

1. [https://github.com/cmangos/issues/wiki/Installation-Instructions](https://github.com/cmangos/issues/wiki/Installation-Instructions)

2. [https://fx.fklds.com/1238.html](https://fx.fklds.com/1238.html)
