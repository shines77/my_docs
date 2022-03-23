# Linux-查看登陆日志-last-lastb命令

## 1. 登陆日志

### 1.1 `last` 命令

显示最近成功登陆的用户信息和日志。

语法：

```shell
last [-num] [-R] [-n num] [-adiowx] [-f file] [-t YYYYMMDDHHMMSS] [name...] [tty...]
```

|   参数  |  长参数 | 描叙      |
|:-------:|-------|----------|
|-a|--hostlast|将登录系统的的主机名称或IP地址，显示在最后一行|
|-d|--dns|将IP地址转换成主机名称|
|-f|--file \<file\>|指定记录文件，默认是显示/var/log目录下的wtmp文件的记录，但/var/log目录下得btmp能显示的内容更丰富，可以显示远程登录，例如ssh登录 ，包括失败的登录请求。|
|-i|--ip|显示特定ip登录的情况。|
|-o||Read an old-type wtmp file (written by linux-libc5 applications).|
|-n|--limit \<number\>|-n <显示行数>或-<显示行数> 　设置显示多少行记录|
|-w|--fullnames|Display full user and domain names in the output|
|-R|--nohostname|不显示登入系统的主机名称或IP（省略 hostname 的栏位）|
|-s|--since \<time\>|显示YYYYMMDDHHMMSS开始的信息|
|-t|--until \<time\>|显示YYYYMMDDHHMMSS之前的信息|
|-x|--sytem|显示系统关闭、用户登录和退出的历史|
|...........|for.too.long.name|....................................|

示例:

```shell
root     pts/4        124.240.25.139   Thu Mar 24 03:51   still logged in
root     pts/4        124.240.25.139   Wed Mar 23 21:48 - 03:51  (06:02)
root     pts/2        124.240.25.139   Wed Mar 23 21:47   still logged in
root     pts/0        124.240.25.139   Wed Mar 23 21:36   still logged in
root     pts/0        124.240.25.139   Tue Mar 22 17:22 - 01:01  (07:38)
root     tty1                          Tue Mar 22 17:02   still logged in
root     pts/2        124.240.25.139   Tue Mar 22 15:45 - 21:47 (1+06:02)
root     pts/0        124.240.25.139   Tue Mar 22 15:44 - 17:22  (01:37)
root     pts/0        124.240.25.139   Mon Mar 21 08:42 - 09:23  (00:41)
root     pts/1        124.240.25.139   Mon Mar 21 07:44 - 13:11  (05:26)
ubuntu   pts/0        124.240.25.139   Mon Mar 21 07:43 - 07:44  (00:01)
reboot   system boot  5.4.0-96-generic Mon Mar 21 07:42   still running
ubuntu   pts/0        106.55.203.38    Mon Mar 21 07:34 - down   (00:07)
```

### 1.2 `lastb` 命令

显示最近未成功登陆的用户信息和日志。

语法：

```shell
last [-R] [-num] [ -n num ] [-adiowx] [ -f file ] [ -t YYYYMMDDHHMMSS ] [name...]  [tty...]
```

|   参数  |  长参数 | 描叙      |
|:-------:|-------|----------|
|-a|--hostlast|将登录系统的的主机名称或IP地址，显示在最后一行|
|-d|--dns|将IP地址转换成主机名称|
|-f|--file \<file\>|指定记录文件，默认是显示/var/log目录下的wtmp文件的记录，但/var/log目录下得btmp能显示的内容更丰富，可以显示远程登录，例如ssh登录 ，包括失败的登录请求。|
|-i|--ip|显示特定ip登录的情况。|
|-o||Read an old-type wtmp file (written by linux-libc5 applications).|
|-n|--limit \<number\>|-n <显示行数>或-<显示行数> 　设置显示多少行记录|
|-w|--fullnames|Display full user and domain names in the output|
|-R|--nohostname|不显示登入系统的主机名称或IP（省略 hostname 的栏位）|
|-s|--since \<time\>|显示YYYYMMDDHHMMSS开始的信息|
|-t|--until \<time\>|显示YYYYMMDDHHMMSS之前的信息|
|-x|--sytem|显示系统关闭、用户登录和退出的历史|
|...........|for.too.long.name|....................................|

示例:

```shell
root     ssh:notty    116.110.77.85    Thu Mar 24 04:27 - 04:27  (00:00)
admin    ssh:notty    116.105.164.150  Thu Mar 24 04:25 - 04:25  (00:00)
admin    ssh:notty    116.105.164.150  Thu Mar 24 04:25 - 04:25  (00:00)
admin    ssh:notty    116.105.164.150  Thu Mar 24 04:25 - 04:25  (00:00)
admin    ssh:notty    116.105.164.150  Thu Mar 24 04:25 - 04:25  (00:00)
1234     ssh:notty    116.105.164.150  Thu Mar 24 04:25 - 04:25  (00:00)
1234     ssh:notty    116.105.164.150  Thu Mar 24 04:25 - 04:25  (00:00)
root     ssh:notty    116.110.77.85    Thu Mar 24 04:24 - 04:24  (00:00)
git      ssh:notty    116.105.164.150  Thu Mar 24 04:24 - 04:24  (00:00)
root     ssh:notty    116.105.164.150  Thu Mar 24 04:24 - 04:24  (00:00)
```

### 1.3 `who` 命令

显示当前在线的用户列表。

示例:

```shell
root     tty1         2022-03-22 17:02
root     pts/0        2022-03-23 21:36 (124.240.25.139)
root     pts/1        2022-03-21 19:14 (124.240.25.139:S.0)
root     pts/4        2022-03-24 03:51 (124.240.25.139)
```