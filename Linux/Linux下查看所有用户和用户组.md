
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
    uucp:x:10:10:uucp:/var/spool/uucp:/usr/sbin/nologin
    proxy:x:13:13:proxy:/bin:/usr/sbin/nologin
    www-data:x:33:33:www-data:/var/www:/usr/sbin/nologin
    backup:x:34:34:backup:/var/backups:/usr/sbin/nologin
    list:x:38:38:Mailing List Manager:/var/list:/usr/sbin/nologin
    irc:x:39:39:ircd:/var/run/ircd:/usr/sbin/nologin
    gnats:x:41:41:Gnats Bug-Reporting System (admin):/var/lib/gnats:/usr/sbin/nologin
    nobody:x:65534:65534:nobody:/nonexistent:/usr/sbin/nologin
    libuuid:x:100:101::/var/lib/libuuid:
    syslog:x:101:104::/home/syslog:/bin/false
    messagebus:x:102:106::/var/run/dbus:/bin/false
    landscape:x:103:109::/var/lib/landscape:/bin/false
    sshd:x:104:65534::/var/run/sshd:/usr/sbin/nologin
    skyinno:x:1000:1000:skyinno,,,:/home/skyinno:/bin/bash
    postgres:x:106:114:PostgreSQL administrator,,,:/var/lib/postgresql:/bin/bash
    mysql:x:107:116:MySQL Server,,,:/nonexistent:/bin/false
    dnsmasq:x:108:65534:dnsmasq,,,:/var/lib/misc:/bin/false
    libvirt-qemu:x:109:117:Libvirt Qemu,,,:/var/lib/libvirt:/bin/false
    libvirt-dnsmasq:x:110:118:Libvirt Dnsmasq,,,:/var/lib/libvirt/dnsmasq:/bin/false
    colord:x:111:120:colord colour management daemon,,,:/var/lib/colord:/bin/false