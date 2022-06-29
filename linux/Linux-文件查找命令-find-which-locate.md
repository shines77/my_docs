# Linux 文件查找命令 find, which, locate

## 1. 前言

`Linux` 系统中查找文件的命令有 `which`、`whereis`、`locate` 和 `find` 等，其中 `find` 命令最为强大。本文对这四条命令进行简单的介绍、列举了一些简单的使用方式。

## 2. which

作用：在 `PATH` 变量中定义的全部路径中查找可执行文件、脚本或文件，并且返回第一个搜索结果。

命令参数：

在 `Ubuntu` 上，`which` 命令只有一个参数，就是 `-a`，因为 `/usr/bin/which` 在 Ubuntu 上只是一个 `bash` 脚本：

* `-a`

    默认情况下，`which` 命令会在匹配到第一个结果后结束运行，添加该参数可以让其搜索所有匹配的结果。

在 `CentOS` 中，有如下参数：

* `-all`, `-a`

    默认情况下，`which` 命令会在匹配到第一个结果后结束运行，添加该参数可以让其搜索所有匹配的结果。

* `-read-alias`, `-i`

    将输入视为别名搜索。

    `Linux` 系统中通常会使用 `alias` 设置诸多别名来简写命令，例如 `CentOS` 中的 `ll` 实际是 `ls -l` ，而 `which` 是 `alias | /usr/bin/which --tty-only --read-alias --show-dot --show-tilde` 。

* `--tty-only`

    尽在终端调用的情况下附带右侧添加的参数，其他情况下不接收右侧其他参数（此处的参数值 `--show-dot`、`--show-tilde` 此类，输入的待查询命令仍然会接收），通过这个命令可以保证 `Shell` 脚本中的 `which` 命令正确执行。

* `--show-dot`

    输出以 “`.`” 符号开头的目录。Linux 中 “`.`” 符号开头的目录是约定的隐藏文件夹，没有该参数时会忽略这些目录。

* `--show-tilde`

    将用户的 `home` 目录替换成 “`~`” 符号输出。`Linux` 中 “`~`” 符号是登录用户 `home` 目录的缩写，如果登录用户名为 `cncsl`，则 “~” 表示 “/home/cncsl” 目录。当使用 `root` 账号登录时，该参数无效。

在其他 `Linux` 版本中，可能有如下参数：

* `-n` : 指定文件名长度，指定的长度必须大于或等于所有文件中最长的文件名。
* `-p` : 与 `-n` 参数相同，但此处的包括了文件的路径。
* `-w` : 指定输出时栏位的宽度。
* `-V` : 显示版本信息。

## 3. whereis

作用：查找指定命令的可执行文件、源代码和手册的位置。

```shell
$ whereis vim

vim: /usr/bin/vim /usr/share/vim /usr/share/man/man1/vim.1.gz
```

从这个命令可以得知，`vim` 的可执行程序位于 `/usr/bin/vim`，手册位于 `/usr/share/vim` 和 `/share/man/man1/vim.1.gz` 目录。

命令参数：

* `-b`、`-m` 和 `-s`

    分别用于指定仅查询可执行文件、手册和源代码。

* `-B`、`-M` 和 `-S`

    用于指定仅查询路径。

* `-u`

    参数的描述直译为 "`仅查询有异常情况的命令`"。所谓的异常情况是指，某个命令的匹配的相关类型文件不是恰好一份（一份都没有或多于一份）。

    例如：

    `ls` 命令具有两份手册：

    ```shell
    $ whereis -m -u ls

    ls: /usr/share/man/man1/ls.1.gz /usr/share/man/man1p/ls.1p.gz
    ```

    `Linux` 系统中有很多个与 `python` 相关的可执行文件：

    ```shell
    $ whereis -b -u python

    python: /usr/bin/python /usr/bin/python2.7 /usr/lib/python2.7 /usr/lib64/python2.7 /etc/python /usr/include/python2.7
    ```

## 4. locate

作用：在文档和目录名称的数据库中查找指定文件。

`Linux` 系统会定期自动扫描磁盘来维护一个记录磁盘数据的数据库，而 `locate` 命令使用的数据库是 `/var/lib/mlocate/mlocate.db` 。

```shell
$ ls -hl /var/lib/mlocate/mlocate.db

-rw-r-----. 1 root slocate 2.7M Feb  4 03:42 /var/lib/mlocate/mlocate.db
```

可以看到，当前 `mlocate.db` 文件共记录了 `2.7M` 的数据。

命令参数：

* `--count`, `-c`

    不输出具体的文件路径信息，仅输出查询到的数量。

* `--ignore-case`, -`i`

    查询时忽略大小写

* `--limit`, `-l`, `-n LIMIT`

    限定输出的文件数量为 `LIMIT` 。

* `--regexp`, `-r REGEXP`

    使用 `REGEXP` 指定的正则表达式匹配。

范例：

```bash
# 统计有多少PNG格式的图像文件
$ locate -c png

# 统计有多少 readme 文件（根据编写者的习惯，readme 文件可能名为 README、ReadMe等）
$ locate -c -i readme

# 输出十个 .gz 归档文件的路径
$ locate -l 10 *.gz

# 查看 tomcat 2021年1月的日志
$ locate -r tomcat.2021-01-[0-3][0-9].log
```

由于 `locate` 命令是从数据库查找文件，新创建的文件可能由于未被记录到数据库中而无法查询到，这种时候需要使用 `updatedb` 命令手动更新数据库。

## 5. find

作用：在一个目录层级中（当前路径或指定的路径）查找文件。

`find` 命令功能强大，可根据多种条件查询文件，随后进行自定义的操作。

命令格式：

```shell
find [path...] [expression] [options]
```
参数说明：

`find` 根据下列规则判断 `path` 和 `expression`，在命令列上第一个 `- ( ) , !` 之前的部份为 `path`，之后的是 `expression`。如果 `path` 是空字符串，则使用目前路径。如果 `expression` 是空字符串，则使用 `-print` 为预设 `expression` 。

`expression` 中可使用的选项有二三十个之多，在此只介绍最常用的部份。

* `-mount`, `-xdev` : 只检查和指定目录在同一个文件系统下的文件，避免列出其它文件系统中的文件

* `-amin n` : 在过去 n 分钟内被读取过

* `-anewer file` : 比文件 file 更晚被读取过的文件

* `-atime n` : 在过去 n 天内被读取过的文件

* `-cmin n` : 在过去 n 分钟内被修改过

* `-cnewer file` :比文件 file 更新的文件

* `-ctime n` : 在过去 n 天内创建的文件

* `-mtime n` : 在过去 n 天内修改过的文件

* `-empty` : 空的文件-gid n or -group name : gid 是 n 或是 group 名称是 name

* `-ipath p`, `-path p` : 路径名称符合 p 的文件，ipath 会忽略大小写

* `-name name`, `-iname name` : 文件名称符合 name 的文件。iname 会忽略大小写

* `-size n` : 文件大小 是 n 单位，b 代表 512 位元组的区块，c 表示字元数，k 表示 kilo bytes，w 是二个位元组。

* `-type c` :

    文件类型是 `c` 的文件。

  * `d` : 目录
  * `c` : 字型装置文件
  * `b` : 区块装置文件
  * `p` : 具名贮列
  * `f` : 一般文件
  * `l` : 符号连结
  * `s` : socket
  * `-pid n` : process id 是 n 的文件

命令参数：

* `-H` :
* `-L` :
* `-print` : 输出文件名
* `-fprint FILE` : 打印文件内容
* `-exec COMMAND` : 运行命令
* `-execdir COMMAND`: 指定运行目录
* `-delete` : 删除匹配的文件
* `-depth` : 指定搜索的深度
* `-mindepth` LEVELS : 指定搜索的最小深度
* `-maxdepth` LEVELS : 指定搜索的最大深度
* `-noleaf` :
* `--help`: 显示帮助信息
* `--version`: 显示版本号

更多信息请使用 `find --help` 查询。

范例：

```bash
# 查询当前目录下所有的 markdown 文档
$ find . -name "*.md"

# 查询当前目录下所有的 log 文档
$ find . -name "*.log"

# 查询用户视频文件夹中大于 100M 的文件
$ find ~/Videos/ -size +100M

# 查询用户音乐文件夹中过去七天访问过的文件
$ find ~/Music/ -atime -7

# 查询系统中、三个月之前创建的、一个月之内没有访问过、大于 30M 的日志文件，并删除
$ find / -ctime +90 -atime +30 -size +1M -name "*.log" -delete

# 查找 /var/log 目录中更改时间在 7 日以前的普通文件，并在删除之前询问它们
$ find /var/log -type f -mtime +7 -ok rm {} \;

# 查找当前目录中文件属主具有读、写权限，并且文件所属组的用户和其他用户具有读权限的文件
$ find . -type f -perm 644 -exec ls -l {} \;

# 查找系统中所有文件长度为 0 的普通文件，并列出它们的完整路径
$ find / -type f -size 0 -exec ls -l {} \;
```

`find` 会实际的扫描磁盘，所以速度会明显慢于前三个命令。

## 6. 参考文章

1. `[Linux 查找文件的正确姿势]`

    [https://cncsl.github.io/2021/0204/Linux%E6%9F%A5%E6%89%BE%E6%96%87%E4%BB%B6%E7%9A%84%E6%AD%A3%E7%A1%AE%E5%A7%BF%E5%8A%BF/](https://cncsl.github.io/2021/0204/Linux%E6%9F%A5%E6%89%BE%E6%96%87%E4%BB%B6%E7%9A%84%E6%AD%A3%E7%A1%AE%E5%A7%BF%E5%8A%BF/)

2. `[Linux find 命令]`

    [https://www.runoob.com/linux/linux-comm-find.html](https://www.runoob.com/linux/linux-comm-find.html)
