
Linux下查看所有用户和用户组
----------------------------

## 当前用户和用户组 ##

* groups 查看当前登录用户的组内成员。
* whoami 查看当前登录用户名。

例如：

    $ groups gliethttp   # 查看 gliethttp 用户所在的组，以及组内成员。

## 查看系统文件 ##

其中：

文件 `/etc/group` 包含所有用户组

文件 `/etc/shadow` 和 `/etc/passwd` 包含系统存在的所有用户

例如：

    $ sudo vim /etc/passwd

内容如下：

    root:x:0:0:root:/root:/bin/bash
    daemon:x:1:1:daemon:/usr/sbin:/usr/sbin/nologin
    bin:x:2:2:bin:/bin:/usr/sbin/nologin
    sys:x:3:3:sys:/dev:/usr/sbin/nologin
    sync:x:4:65534:sync:/bin:/bin/sync
    games:x:5:60:games:/usr/games:/usr/sbin/nologin
    man:x:6:12:man:/var/cache/man:/usr/sbin/nologin
    lp:x:7:7:lp:/var/spool/lpd:/usr/sbin/nologin
    mail:x:8:8:mail:/var/mail:/usr/sbin/nologin
    news:x:9:9:news:/var/spool/news:/usr/sbin/nologin
    .................

## 通过命令显示 ##

    # 注意：最后是一个 “：” 冒号，它是分隔符 (delimiter)

    $ cat /etc/passwd | cut -f 1 -d :

    root
    daemon
    bin
    sys
    sync
    games
    man
    lp
    mail
    news
    uucp
    proxy
    www-data
    backup
    ......

