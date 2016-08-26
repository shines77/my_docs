
1）查找被占用的端口

```shell

    $ netstat -tlnp
    $ netstat -tlnp | grep 8083

    $ netstat -tlnp 查看端口使用情况，而 netstat -tln | grep 8083 则是只查看端口8083的使用情况.

    $ netstat -tunlp | grep 22

    $ netstat -anp 显示系统端口使用情况

    $ netstat -lnp 较简单的参数
```

2）查看端口属于哪个程序？端口被哪个进程占用

```shell

    $ lsof -i:8083

    $ lsof -i:21
```

3）Socket Statistics

ss 是 Socket Statistics 的缩写。顾名思义，ss 命令可以用来获取 socket 统计信息，它可以显示和 netstat 类似的内容。但 ss 的优势在于它能够显示更多更详细的有关 TCP 和连接状态的信息，而且比 netstat 更快速更高效。

(CentOS 里要先安装 IPRoute2)

命令参数：

```shell

    -h, --help	    帮助信息
    -V, --version	程序版本信息
    -n, --numeric	不解析服务名称
    -r, --resolve   解析主机名
    -a, --all	    显示所有套接字（sockets）
    -l, --listening	显示监听状态的套接字（sockets）
    -o, --options   显示计时器信息
    -e, --extended  显示详细的套接字（sockets）信息
    -m, --memory    显示套接字（socket）的内存使用情况
    -p, --processes	显示使用套接字（socket）的进程
    -i, --info	    显示 TCP内部信息
    -s, --summary	显示套接字（socket）使用概况
    -4, --ipv4      仅显示IPv4的套接字（sockets）
    -6, --ipv6      仅显示IPv6的套接字（sockets）
    -0, --packet	显示 PACKET 套接字（socket）
    -t, --tcp	    仅显示 TCP套接字（sockets）
    -u, --udp	    仅显示 UCP套接字（sockets）
    -d, --dccp	    仅显示 DCCP套接字（sockets）
    -w, --raw	    仅显示 RAW套接字（sockets）
    -x, --unix	    仅显示 Unix套接字（sockets）
    -f, --family=FAMILY  显示 FAMILY类型的套接字（sockets），FAMILY可选，支持  unix, inet, inet6, link, netlink
    -A, --query=QUERY, --socket=QUERY
        QUERY := {all|inet|tcp|udp|raw|unix|packet|netlink}[,QUERY]
    -D, --diag=FILE     将原始TCP套接字（sockets）信息转储到文件
    -F, --filter=FILE  从文件中都去过滤器信息
        FILTER := [ state TCP-STATE ] [ EXPRESSION ]
```

实例1：显示TCP连接

```shell

    $ ss -t -a
    
    State      Recv-Q Send-Q       Local Address:Port          Peer Address:Port
    LISTEN     0      0                127.0.0.1:smux                     *:*
    LISTEN     0      0                        *:3690                     *:*
    LISTEN     0      0                        *:ssh                      *:*
    ESTAB      0      0          192.168.120.204:ssh              10.2.0.68:49368
```

实例2：显示 Sockets 摘要

```shell

    $ ss -s
    
    Total: 34 (kernel 48)
    TCP:   4 (estab 1, closed 0, orphaned 0, synrecv 0, timewait 0/0), ports 3
    
    Transport Total     IP        IPv6
    *         48        -         -
    RAW       0         0         0
    UDP       5         5         0
    TCP       4         4         0
    INET      9         9         0
    FRAG      0         0         0
```

实例3：列出所有打开的网络连接端口

```shell

    $ ss -l
    
    Recv-Q Send-Q       Local Address:Port           Peer Address:Port
    0      0                127.0.0.1:smux                      *:*
    0      0                        *:3690                      *:*
    0      0                        *:ssh                       *:*
```

实例4：查看进程使用的socket

```shell

    $ ss -pl

```

实例5：找出打开套接字/端口应用程序

```shell

    $ ss -lp | grep 3306

```

实例6：显示所有UDP Sockets

```shell

    $ ss -u -a

```

实例7：显示所有状态为Established的HTTP连接

```shell

    $ ss -o state established '( dport = :http or sport = :http )'

```

See: [http://www.cnblogs.com/peida/archive/2013/03/11/2953420.html](http://www.cnblogs.com/peida/archive/2013/03/11/2953420.html)

4）杀掉占用端口的进程

    kill -9 进程id
