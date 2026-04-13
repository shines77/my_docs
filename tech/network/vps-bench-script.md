# VPS benchmark script

## VPS benchmark

### 1. `NodeQuality@LloydAsp`

在沙箱环境中运行 vps 测试脚本 (Hardware, IP, Network)，并排版测试结果。

脚本基于 [https://github.com/xykt](https://github.com/xykt) 的相关项目改写，见下文。

官网：[https://NodeQuality.com](https://NodeQuality.com)

GitHub：[https://github.com/LloydAsp/NodeQuality](https://github.com/LloydAsp/NodeQuality)

运行命令：

```bash
bash <(curl -sL https://run.NodeQuality.com)
```

### 2. `Hardware@xykt`

硬件质量体检脚本 - Hardware Quality Check Script

GitHub：[https://github.com/xykt/HardwareQuality](https://github.com/xykt/HardwareQuality)

运行命令：

```bash
bash <(curl -sL https://Check.Place) -H
```

可选模式有：

- 标准模式
- 快速模式
- 硬盘模式
- 深度模式

### 3. `IPQuality@xykt`

IP质量检测脚本 - IP Quality Check Script

GitHub：[https://github.com/xykt/IPQuality](https://github.com/xykt/IPQuality)

运行命令：

```bash
bash <(curl -sL https://Check.Place) -I
```

默认双线模式：

```bash
bash <(curl -sL https://IP.Check.Place)
```

只检测 `ipv4` 结果：

```bash
bash <(curl -sL https://IP.Check.Place) -4
```

只检测 `ipv6` 结果：

```bash
bash <(curl -sL https://IP.Check.Place) -6
```

只检测指定网卡：

```bash
bash <(curl -sL https://IP.Check.Place) -i eth0
```

### 4. `NetQuality@xykt`

网络质量检测脚本 - Network Quality Check Script

GitHub：[https://github.com/xykt/NetQuality](https://github.com/xykt/NetQuality)

运行命令：

```bash
bash <(curl -sL https://Net.Check.Place)
```

便捷模式可选的模式有：

- 完整模式
- 省流模式 (<100MB)
- 三网完整路由
- 三网延迟模式
- 自定义精简模式

只检测 `ipv4` 结果：

```bash
bash <(curl -sL https://Net.Check.Place) -4
```

只检测 `ipv6` 结果：

```bash
bash <(curl -sL https://Net.Check.Place) -6
```

低测试数据模式（省流）：

```bash
bash <(curl -sL https://Net.Check.Place) -L
```

缺省状态，默认只检测北京、上海、广东三地的路由：

```bash
bash <(curl -sL https://Net.Check.Place) -R
```

完整路由模式（TCP大包）：

```bash
bash <(curl -sL https://Net.Check.Place) -R [大陆地区省级行政区名称或中/英文简称]

# 例如：
bash <(curl -sL https://Net.Check.Place) -R 桂
bash <(curl -sL https://Net.Check.Place) -R 广西
bash <(curl -sL https://Net.Check.Place) -R GX
bash <(curl -sL https://Net.Check.Place) -R gx
```

只检测延迟模式：

```bash
bash <(curl -sL https://Net.Check.Place) -P
```

## 较早之前的 VPS 测试脚本

### 1. bench.sh

[bench.sh](https://bench.sh/) 是一个 Linux 系统性能基准测试工具。它的测试结果如下图：给出服务器的整体配置信息，IO 性能，网络性能。很多人使用它测试 vps 性能。

服务器在国外（因为可能需要科学上网），可以使用以下命令运行测试：

```bash
wget -qO- bench.sh | bash
```

​服务器在国内，则只能从 [https://github.com/teddysun/across/blob/master/bench.sh](https://github.com/teddysun/across/blob/master/bench.sh) 复制脚本到本地运行，保存脚本文件为 `bench.sh`，使用`chmod +x` 添加运行权限, 使用 `./bench.sh` 命令运行。

测试范例：

```bash
-------------------- A Bench.sh Script By Teddysun -------------------
 Version            : v2026-01-31
 Usage              : wget -qO- bench.sh | bash
----------------------------------------------------------------------
 CPU Model          : Intel(R) Xeon(R) CPU E5-2680 v2 @ 2.80GHz
 CPU Cores          : 2 @ 2799.998 MHz
 CPU Cache          : 25600 KB
 AES-NI             : ✓ Enabled
 VM-x/AMD-V         : ✓ Enabled
 Total Disk         : 48.4 GB (4.2 GB Used)
 Total RAM          : 961.1 MB (358.3 MB Used)
 System Uptime      : 6 days, 19 hour 38 min
 Load Average       : 0.00, 0.00, 0.00
 OS                 : Ubuntu 24.04.4 LTS
 Arch               : x86_64 (64 Bit)
 Kernel             : 6.8.0-106-generic
 TCP Congestion Ctrl: cubic
 Virtualization     : KVM
 IPv4/IPv6          : ✓ Online / ✗ Offline
 Organization       : AS40004 CONRADIT, LLC
 Location           : Hong Kong / HK
 Region             : Hong Kong
----------------------------------------------------------------------
 I/O Speed(1st run) : 423 MB/s
 I/O Speed(2nd run) : 438 MB/s
 I/O Speed(3rd run) : 423 MB/s
 I/O Speed(average) : 428.0 MB/s
----------------------------------------------------------------------
 Node Name        Upload Speed      Download Speed      Latency     
 Speedtest.net    Test failed       
 Los Angeles, US  10.49 Mbps        48.41 Mbps          150.02 ms   
 Dallas, US       10.52 Mbps        48.45 Mbps          185.58 ms   
 Montreal, CA     10.18 Mbps        48.19 Mbps          218.27 ms   
 Paris, FR        10.00 Mbps        48.01 Mbps          190.86 ms   
 Amsterdam, NL    10.15 Mbps        49.45 Mbps          255.01 ms   
 Suzhou, CN       10.52 Mbps        49.27 Mbps          158.39 ms   
 Ningbo, CN       5.95 Mbps         49.40 Mbps          54.14 ms    
 Hong Kong, CN    9.90 Mbps         48.99 Mbps          3.44 ms     
 Singapore, SG    10.26 Mbps        48.68 Mbps          41.25 ms    
 Taipei, CN       10.24 Mbps        48.62 Mbps          15.10 ms    
 Tokyo, JP        10.77 Mbps        48.59 Mbps          157.98 ms   
----------------------------------------------------------------------
 Finished in        : 5 min 24 sec
 Timestamp          : 2026-03-27 13:09:46 UTC
----------------------------------------------------------------------
```
