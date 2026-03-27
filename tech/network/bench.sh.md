# bench.sh：一行命令测试Linux服务器基准性能

## 简介

[bench.sh](https://bench.sh/) 是一个 Linux 系统性能基准测试工具。它的测试结果如下图：给出服务器的整体配置信息，IO 性能，网络性能。很多人使用它测试 vps 性能。

## 一键运行

服务器在国外可以使用以下命令运行测试：

```bash
wget -qO- bench.sh | bash
```

​服务器在国内只能从 [github](https://github.com/teddysun/across/blob/master/bench.sh) 复制脚本到本地运行。脚本内容已经贴到下面：复制代码块，保存到 bench.sh 文件中，chmod +x 赋权, ./bench.sh 运行。

复制脚本到本地运行：

```bash
#!/usr/bin/env bash
#
# Description: A Bench Script by Teddysun
#
# Copyright (C) 2015 - 2023 Teddysun <i@teddysun.com>
# Thanks: LookBack <admin@dwhd.org>
# URL: https://teddysun.com/444.html
# https://github.com/teddysun/across/blob/master/bench.sh
#
trap _exit INT QUIT TERM

_red() {
   
   
    printf '\033[0;31;31m%b\033[0m' "$1"
}

_green() {
   
   
    printf '\033[0;31;32m%b\033[0m' "$1"
}

_yellow() {
   
   
    printf '\033[0;31;33m%b\033[0m' "$1"
}

_blue() {
   
   
    printf '\033[0;31;36m%b\033[0m' "$1"
}

_exists() {
   
   
    local cmd="$1"
    if eval type type >/dev/null 2>&1; then
        eval type "$cmd" >/dev/null 2>&1
    elif command >/dev/null 2>&1; then
        command -v "$cmd" >/dev/null 2>&1
    else
        which "$cmd" >/dev/null 2>&1
    fi
    local rt=$?
    return ${rt}
}

_exit() {
   
   
    _red "\nThe script has been terminated. Cleaning up files...\n"
    # clean up
    rm -fr speedtest.tgz speedtest-cli benchtest_*
    exit 1
}

get_opsy() {
   
   
    [ -f /etc/redhat-release ] && awk '{print $0}' /etc/redhat-release && return
    [ -f /etc/os-release ] && awk -F'[= "]' '/PRETTY_NAME/{print $3,$4,$5}' /etc/os-release && return
    [ -f /etc/lsb-release ] && awk -F'[="]+' '/DESCRIPTION/{print $2}' /etc/lsb-release && return
}

next() {
   
   
    printf "%-70s\n" "-" | sed 's/\s/-/g'
}

speed_test() {
   
   
    local nodeName="$2"
    if [ -z "$1" ];then
        ./speedtest-cli/speedtest --progress=no --accept-license --accept-gdpr >./speedtest-cli/speedtest.log 2>&1
    else
        ./speedtest-cli/speedtest --progress=no --server-id="$1" --accept-license --accept-gdpr >./speedtest-cli/speedtest.log 2>&1
    fi
    if [ $? -eq 0 ]; then
        local dl_speed up_speed latency
        dl_speed=$(awk '/Download/{print $3" "$4}' ./speedtest-cli/speedtest.log)
        up_speed=$(awk '/Upload/{print $3" "$4}' ./speedtest-cli/speedtest.log)
        latency=$(awk '/Latency/{print $3" "$4}' ./speedtest-cli/speedtest.log)
        if [[ -n "${dl_speed}" && -n "${up_speed}" && -n "${latency}" ]]; then
            printf "\033[0;33m%-18s\033[0;32m%-18s\033[0;31m%-20s\033[0;36m%-12s\033[0m\n" " ${nodeName}" "${up_speed}" "${dl_speed}" "${latency}"
        fi
    fi
}

speed() {
   
   
    speed_test '' 'Speedtest.net'
    speed_test '21541' 'Los Angeles, US'
    speed_test '43860' 'Dallas, US'
    speed_test '40879' 'Montreal, CA'
    speed_test '24215' 'Paris, FR'
    speed_test '28922' 'Amsterdam, NL'
    speed_test '24447' 'Shanghai, CN'
    speed_test '5530' 'Chongqing, CN'
    speed_test '60572' 'Guangzhou, CN'
    speed_test '32155' 'Hongkong, CN'
    speed_test '23647' 'Mumbai, IN'
    speed_test '13623' 'Singapore, SG'
    speed_test '21569' 'Tokyo, JP'
}

io_test() {
   
   
    (LANG=C dd if=/dev/zero of=benchtest_$$ bs=512k count="$1" conv=fdatasync && rm -f benchtest_$$) 2>&1 | awk -F '[,，]' '{io=$NF} END { print io}' | sed 's/^[ \t]*//;s/[ \t]*$//'
}

calc_size() {
   
   
    local raw=$1
    local total_size=0
    local num=1
    local unit="KB"
    if ! [[ ${raw} =~ ^[0-9]+$ 
```

## 参考文章

- [bench.sh：一行命令测试Linux服务器基准性能](https://blog.csdn.net/qq_38641599/article/details/142580250)
