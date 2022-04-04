# Linux benchmark tools

## 1. 性能测试脚本

### 1.1 Bench.sh

`https://bench.sh` - view system information and test the network, disk of your Linux server.

命令：

```shell
wget -qO- bench.sh | bash

或

wget -c http://bench.sh -O bench.sh && sudo chmod +x bench.sh && ./bench.sh
```

测试效果：

```text
-------------------- A Bench.sh Script By Teddysun -------------------
 Version            : v2022-02-22
 Usage              : wget -qO- bench.sh | bash
----------------------------------------------------------------------
 CPU Model          : AMD EPYC 7K62 48-Core Processor
 CPU Cores          : 2 @ 2595.124 MHz
 CPU Cache          : 512 KB
 AES-NI             : Enabled
 VM-x/AMD-V         : Disabled
 Total Disk         : 78.7 GB (13.1 GB Used)
 Total Mem          : 3.7 GB (204.6 MB Used)
 System uptime      : 11 days, 3 hour 49 min
 Load average       : 0.05, 0.09, 0.22
 OS                 : Ubuntu 18.04.6 LTS
 Arch               : x86_64 (64 Bit)
 Kernel             : 4.15.0-142-generic
 TCP CC             : cubic
 Virtualization     : KVM
 Organization       : AS45090 Shenzhen Tencent Computer Systems Company Limited
 Location           : Beijing / CN
 Region             : Beijing
----------------------------------------------------------------------
 I/O Speed(1st run) : 159 MB/s
 I/O Speed(2nd run) : 144 MB/s
 I/O Speed(3rd run) : 144 MB/s
 I/O Speed(average) : 149.0 MB/s
----------------------------------------------------------------------
 Node Name        Upload Speed      Download Speed      Latency     
 Speedtest.net    8.10 Mbps         109.72 Mbps         19.49 ms    
 Los Angeles, US  8.98 Mbps         107.98 Mbps         200.85 ms   
 Dallas, US       8.28 Mbps         122.23 Mbps         203.33 ms   
 Montreal, CA     8.95 Mbps         109.40 Mbps         283.29 ms   
 Paris, FR        8.48 Mbps         1.31 Mbps           269.15 ms   
 Amsterdam, NL    8.43 Mbps         74.54 Mbps          254.10 ms   
 Shanghai, CN     8.21 Mbps         102.69 Mbps         30.88 ms    
 Nanjing, CN      8.20 Mbps         104.13 Mbps         25.81 ms    
 Guangzhou, CN    7.95 Mbps         112.83 Mbps         10.91 ms    
 Seoul, KR        8.38 Mbps         123.16 Mbps         104.57 ms   
----------------------------------------------------------------------
```

结论：挺不错的（推荐），唯一的缺点就是没有内存的测试。

### 1.2 Linux Bench

`https://github.com/STH-Dev/linux-bench` - `Linux-Bench` is a script that runs hardinfo, `Unixbench 5.1.3`, `c-ray 1.1`, `STREAM`, `OpenSSL`, `sysbench` (CPU), `crafty`, `redis`, `NPB`, `NAMD`, and `7-zip` benchmarks.

命令：

```shell
wget https://raw.githubusercontent.com/STH-Dev/linux-bench/master/linux-bench.sh && sudo chmod +x linux-bench.sh && ./linux-bench.sh
```

运行效果：

```text
   #    #  #    #  #  #    #          #####   ######  #    #   ####   #    #
   #    #  ##   #  #   #  #           #    #  #       ##   #  #    #  #    #
   #    #  # #  #  #    ##            #####   #####   # #  #  #       ######
   #    #  #  # #  #    ##            #    #  #       #  # #  #       #    #
   #    #  #   ##  #   #  #           #    #  #       #   ##  #    #  #    #
    ####   #    #  #  #    #          #####   ######  #    #   ####   #    #

   Version 5.1.3                      Based on the Byte Magazine Unix Benchmark

   Multi-CPU version                  Version 5 revisions by Ian Smith,
                                      Sunnyvale, CA, USA
   January 13, 2011                   johantheghost at yahoo period com


1 x Dhrystone 2 using register variables  1 2 3 4 5 6 7 8 9 10

1 x Double-Precision Whetstone  1 2 3 4 5 6 7 8 9 10

1 x System Call Overhead  1 2 3 4 5 6 7 8 9 10

...........

------------------------------------------------------------------------
Benchmark Run: Mon Sep 02 2019 10:50:09 - 11:18:44
4 CPUs in system; running 4 parallel copies of tests

Dhrystone 2 using register variables       85510590.5 lps   (10.0 s, 7 samples)
Double-Precision Whetstone                    14574.0 MWIPS (11.2 s, 7 samples)
Execl Throughput                               7195.1 lps   (30.0 s, 2 samples)
File Copy 1024 bufsize 2000 maxblocks        795781.4 KBps  (30.0 s, 2 samples)
File Copy 256 bufsize 500 maxblocks          239085.0 KBps  (30.0 s, 2 samples)
File Copy 4096 bufsize 8000 maxblocks       2527023.3 KBps  (30.0 s, 2 samples)
Pipe Throughput                             2891575.3 lps   (10.0 s, 7 samples)
Pipe-based Context Switching                 406038.2 lps   (10.0 s, 7 samples)
Process Creation                              14031.2 lps   (30.0 s, 2 samples)
Shell Scripts (1 concurrent)                  12104.2 lpm   (60.0 s, 2 samples)
Shell Scripts (8 concurrent)                   1664.1 lpm   (60.0 s, 2 samples)
System Call Overhead                        2819570.0 lps   (10.0 s, 7 samples)

System Benchmarks Index Values               BASELINE       RESULT    INDEX
Dhrystone 2 using register variables         116700.0   85510590.5   7327.4
Double-Precision Whetstone                       55.0      14574.0   2649.8
Execl Throughput                                 43.0       7195.1   1673.3
File Copy 1024 bufsize 2000 maxblocks          3960.0     795781.4   2009.5
File Copy 256 bufsize 500 maxblocks            1655.0     239085.0   1444.6
File Copy 4096 bufsize 8000 maxblocks          5800.0    2527023.3   4356.9
Pipe Throughput                               12440.0    2891575.3   2324.4
Pipe-based Context Switching                   4000.0     406038.2   1015.1
Process Creation                                126.0      14031.2   1113.6
Shell Scripts (1 concurrent)                     42.4      12104.2   2854.8
Shell Scripts (8 concurrent)                      6.0       1664.1   2773.4
System Call Overhead                          15000.0    2819570.0   1879.7
                                                                   ========
System Benchmarks Index Score                                        2241.3
```

结论：测试过程太漫长了，非常的慢，不推荐。

### 1.3 bench-sh-2

`https://github.com/hidden-refuge/bench-sh-2` - System Info + Speedtest IPv4 + Drive Speed.

原仓库已经被删除，有两个备用的仓库：

`https://github.com/bailus/bench.sh`

`https://github.com/miaocloud/bench-sh-2`

命令：

```shell
wget https://raw.githubusercontent.com/bailus/bench.sh/master/bench.sh -O bench-sh-2.sh && sudo chmod +x bench-sh-2.sh && ./bench-sh-2.sh
```

测试效果：

```text
System Info
-----------
       Processor : Intel(R) Xeon(R) Platinum 8255C CPU @ 2.50GHz
       CPU Cores : 2
       Frequency : 2494.140 MHz
          Memory : 3931 MB
            Swap : 0 MB
          Uptime : up 1 week, 4 days, 1 hour, 41 minutes

              OS : Ubuntu 20.04.4 LTS
            Arch : x86_64 (64 Bit)
          Kernel : 5.4.0-96-generic
        Hostname : VM-0-6-ubuntu


Disk Speed
----------
   I/O (1st run) : 168 MB/s
   I/O (2nd run) : 153 MB/s
   I/O (3rd run) : 157 MB/s
     Average I/O : 159.333 MB/s


IPv4 Speed Test
---------------
  Public address : 123.45.67.89

WARNING: Downloading 10x 100 MB files. 1 GB bandwidth will be used!

    Location            Provider        Speed
    CDN                 Cachefly        10.8MB/s

    Atlanta, GA, US     Coloat          10.6MB/s 
    Dallas, TX, US      Softlayer       11.1MB/s 
    Seattle, WA, US     Softlayer       5.91MB/s 
    San Jose, CA, US    Softlayer       5.74MB/s 
    Washington, DC, US  Softlayer       4.88MB/s 

    Tokyo, Japan        Linode          630KB/s 
    Singapore           Softlayer       2.86MB/s 

    Rotterdam, Netherlands  id3.net     5.57MB/s
    Haarlem, Netherlands    Leaseweb    1.83MB/s
```

结论：因为要下载 100 MB的文件 10 次，非常慢，所以带宽不是很高的 `VPS`，不建议使用。

### 1.4 unixbench.sh

`https://github.com/teddysun/across/blob/master/unixbench.sh` - Auto install `unixbench` and test script. (`https://teddysun.com/245.html`)

命令：

```shell
wget --no-check-certificate https://github.com/teddysun/across/raw/master/unixbench.sh && sudo chmod +x unixbench.sh && ./unixbench.sh
```

没测试，应该跟 `1.2 小节` 的差不多。

## 2. 内存工具

### 2.1 内存带宽测试工具 mbw

官网：[https://github.com/raas/mbw](https://github.com/raas/mbw)。

mbw 是一个内存带宽测试工具，可以测试内存拷贝 memcpy，字符串拷贝 dump，内存块拷贝 mcblock 三种不同的方式下的内存拷贝速度。

支持系统：Ubuntu / CentOS

安装：

```shell
apt-get install mbw
```

如果没有安装源没有可用的版本，也可以自己从 git clone：

```shell
git clone https://github.com/raas/mbw.git
cd mbw
make
```

测试命令：

```shell
# -q 隐藏日志；-n 10 表示运行 10 次，256(MB) 表示测试所用的内存大小
./mbw -q -n 10 256

# 如果是用 apt 安装的，前面不用加 "./"
mbw -q -n 10 256
```

测试结果：

```text
0       Method: MEMCPY  Elapsed: 0.02746        MiB: 256.00000  Copy: 9324.349 MiB/s
1       Method: MEMCPY  Elapsed: 0.02702        MiB: 256.00000  Copy: 9475.515 MiB/s
2       Method: MEMCPY  Elapsed: 0.02725        MiB: 256.00000  Copy: 9395.185 MiB/s
3       Method: MEMCPY  Elapsed: 0.03507        MiB: 256.00000  Copy: 7299.686 MiB/s
4       Method: MEMCPY  Elapsed: 0.02877        MiB: 256.00000  Copy: 8897.849 MiB/s
5       Method: MEMCPY  Elapsed: 0.02903        MiB: 256.00000  Copy: 8819.071 MiB/s
6       Method: MEMCPY  Elapsed: 0.03066        MiB: 256.00000  Copy: 8349.369 MiB/s
7       Method: MEMCPY  Elapsed: 0.02957        MiB: 256.00000  Copy: 8657.423 MiB/s
8       Method: MEMCPY  Elapsed: 0.02806        MiB: 256.00000  Copy: 9123.958 MiB/s
9       Method: MEMCPY  Elapsed: 0.02771        MiB: 256.00000  Copy: 9237.542 MiB/s
AVG     Method: MEMCPY  Elapsed: 0.02906        MiB: 256.00000  Copy: 8809.633 MiB/s
0       Method: DUMB    Elapsed: 0.01714        MiB: 256.00000  Copy: 14932.338 MiB/s
1       Method: DUMB    Elapsed: 0.01709        MiB: 256.00000  Copy: 14979.520 MiB/s
2       Method: DUMB    Elapsed: 0.01803        MiB: 256.00000  Copy: 14196.196 MiB/s
3       Method: DUMB    Elapsed: 0.01739        MiB: 256.00000  Copy: 14724.491 MiB/s
4       Method: DUMB    Elapsed: 0.01734        MiB: 256.00000  Copy: 14765.256 MiB/s
5       Method: DUMB    Elapsed: 0.01724        MiB: 256.00000  Copy: 14844.883 MiB/s
6       Method: DUMB    Elapsed: 0.01697        MiB: 256.00000  Copy: 15089.891 MiB/s
7       Method: DUMB    Elapsed: 0.01736        MiB: 256.00000  Copy: 14746.544 MiB/s
8       Method: DUMB    Elapsed: 0.01709        MiB: 256.00000  Copy: 14982.150 MiB/s
9       Method: DUMB    Elapsed: 0.01723        MiB: 256.00000  Copy: 14862.119 MiB/s
AVG     Method: DUMB    Elapsed: 0.01729        MiB: 256.00000  Copy: 14808.559 MiB/s
0       Method: MCBLOCK Elapsed: 0.01905        MiB: 256.00000  Copy: 13441.142 MiB/s
1       Method: MCBLOCK Elapsed: 0.01939        MiB: 256.00000  Copy: 13204.725 MiB/s
2       Method: MCBLOCK Elapsed: 0.01978        MiB: 256.00000  Copy: 12945.638 MiB/s
3       Method: MCBLOCK Elapsed: 0.01964        MiB: 256.00000  Copy: 13033.296 MiB/s
4       Method: MCBLOCK Elapsed: 0.01986        MiB: 256.00000  Copy: 12887.636 MiB/s
5       Method: MCBLOCK Elapsed: 0.01945        MiB: 256.00000  Copy: 13165.338 MiB/s
6       Method: MCBLOCK Elapsed: 0.01923        MiB: 256.00000  Copy: 13315.302 MiB/s
7       Method: MCBLOCK Elapsed: 0.01970        MiB: 256.00000  Copy: 12997.563 MiB/s
8       Method: MCBLOCK Elapsed: 0.01952        MiB: 256.00000  Copy: 13112.739 MiB/s
9       Method: MCBLOCK Elapsed: 0.01898        MiB: 256.00000  Copy: 13488.593 MiB/s
```

结论：测试速度非常快，至于准确性有待研究，我个人觉得还有提升的空间，不过大致也就这样了。

### 2.2 性能测试工具 lmbench

`lmbench` 是个用于评价系统综合性能的多平台开源 `benchmark`，能够测试包括磁盘读写 I/O 性能、内存读写性能、缓存/内存延迟、进程创建销毁开销、网络性能等。

官网： [http://www.bitmover.com/lmbench/](http://www.bitmover.com/lmbench/)

安装：

```shell
apt-get install lmbench
```

`lmbench` 的手动编译安装方式，请自行百度，这里不再敖述。

运行：

```shell
lmbench-run
```

然后按照提示选择。

测试结果：

```text
There is a mailing list for discussing lmbench hosted at BitMover. 
Send mail to majordomo@bitmover.com to join the list.

Using config in CONFIG.VM-20-17-ubuntu
Mon Apr  4 21:09:40 CST 2022
Latency measurements
Mon Apr  4 21:09:57 CST 2022
Local networking
Mon Apr  4 21:09:59 CST 2022
Bandwidth measurements
Mon Apr  4 21:10:16 CST 2022
Calculating effective TLB size
Mon Apr  4 21:10:18 CST 2022
Calculating memory load parallelism
Mon Apr  4 21:11:22 CST 2022
McCalpin's STREAM benchmark
Mon Apr  4 21:11:24 CST 2022
Calculating memory load latency
Mon Apr  4 21:19:16 CST 2022
Mailing results
./results: 40: ./results: mail: not found
Benchmark run finished....
Remember you can find the results of the benchmark 
under /var/lib/lmbench/results
```

最终的测试结果在 `/var/lib/lmbench/results` 文件夹下面，例如：

```text
/var/lib/lmbench/results/x86_64-linux-gnu/VM-20-17-ubuntu.0
```

测试结果在上面这个文件里，关于内存的数据大概是这样：

```text
STREAM copy latency: 0.77 nanoseconds
STREAM copy bandwidth: 20739.66 MB/sec
STREAM scale latency: 0.72 nanoseconds
STREAM scale bandwidth: 22210.65 MB/sec
STREAM add latency: 0.93 nanoseconds
STREAM add bandwidth: 25829.88 MB/sec
STREAM triad latency: 0.96 nanoseconds
STREAM triad bandwidth: 24902.72 MB/sec
STREAM2 fill latency: 0.59 nanoseconds
STREAM2 fill bandwidth: 13514.94 MB/sec
STREAM2 copy latency: 0.83 nanoseconds
STREAM2 copy bandwidth: 19262.60 MB/sec
STREAM2 daxpy latency: 0.78 nanoseconds
STREAM2 daxpy bandwidth: 30607.36 MB/sec
STREAM2 sum latency: 0.95 nanoseconds
STREAM2 sum bandwidth: 8440.49 MB/sec
```

### 2.3 内存压力测试工具 memtester4

官网： [https://pyropus.ca./software/memtester](https://pyropus.ca./software/memtester)

下载，编译和安装：

```shell
wget -c https://pyropus.ca./software/memtester/old-versions/memtester-4.5.1.tar.gz
sudo tar -xvf memtester-4.5.1.tar.gz
cd nentester-4.5.1
make
make install
```

测试命令：

```shell
命令行格式：
Usage: ./memtester [-p physaddrbase [-d device]] <mem>[B|K|M|G] [loops]

# 测试 256MB 内存，循环两次
memtester 256M 2

# 测试 512MB 内存，循环一次
memtester 512M 1
```

测试结果：

```text
memtester version 4.5.1 (64-bit)
Copyright (C) 2001-2020 Charles Cazabon.
Licensed under the GNU General Public License version 2 (only).

pagesize is 4096
pagesizemask is 0xfffffffffffff000
want 1024MB (1073741824 bytes)
got  1024MB (1073741824 bytes), trying mlock ...locked.
Loop 1/1:
  Stuck Address       : ok         
  Random Value        : ok
  Compare XOR         : ok
  Compare SUB         : ok
  Compare MUL         : ok
  Compare DIV         : ok
  Compare OR          : ok
  Compare AND         : ok
  Sequential Increment: ok
  Solid Bits          : ok         
  Block Sequential    : ok         
  Checkerboard        : ok         
  Bit Spread          : ok         
  Bit Flip            : ok         
  Walking Ones        : ok         
  Walking Zeroes      : ok         
  8-bit Writes        : ok
  16-bit Writes       : ok
```

结论：很可惜，这个工具支持测试内存的运算是否是正常的，并没有性能测试功能。

## 3. 参考文章

- [Linux的bench压测工具和脚本](https://blog.csdn.net/Jailman/article/details/119997971)
- [Linux benchmark scripts and tools](https://haydenjames.io/linux-benchmark-scripts-tools/)
- [Linux系统性能测试工具（一）——内存带宽测试工具mbw](https://www.cnblogs.com/sunshine-blog/p/11903842.html)
