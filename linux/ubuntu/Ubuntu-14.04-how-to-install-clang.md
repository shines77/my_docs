
# How to install clang on Ubuntu 14.04

## 1. 添加 clang 更新源 ##

先在 `Ubuntu 14.04` 上添加 `clang` 的更新源：

1) 备份原来的源文件：

    ```
    $ sudo cp /etc/apt/sources.list /etc/apt/sources.list.old
    ```

2) 编辑源文件

    ```
    $ sudo vim /etc/apt/sources.list
    ```

3) 添加 clang 的源

    `clang` 的 `APT` 源信息在官网：[https://apt.llvm.org/]() （可能需要翻墙），查阅其中的 `Ubuntu` 部分。

    以下是 `Ubuntu 14.04` 的源信息，把它添加到 `/etc/apt/sources.list` 文件里：

    ```
    deb http://apt.llvm.org/trusty/ llvm-toolchain-trusty main
    deb-src http://apt.llvm.org/trusty/ llvm-toolchain-trusty main
    # clang 5.0 
    deb http://apt.llvm.org/trusty/ llvm-toolchain-trusty-5.0 main
    deb-src http://apt.llvm.org/trusty/ llvm-toolchain-trusty-5.0 main
    # clang 6.0 
    deb http://apt.llvm.org/trusty/ llvm-toolchain-trusty-6.0 main
    deb-src http://apt.llvm.org/trusty/ llvm-toolchain-trusty-6.0 main

    # Also add the following for the appropriate libstdc++
    # deb http://ppa.launchpad.net/ubuntu-toolchain-r/test/ubuntu trusty main
    ```

    其他 `Linux` 发行版本的源信息可查阅上面的官网。 

    注：`libc++` 是针对 `clang` 特别重写的 `C++` 标准库，也算是 `clang` 的 “御用” 库了。这就像 `libstdc++` 和 `gcc` 的关系一样，但 `clang` 也可以使用 `libstdc++`。如果想使用 `libstdc++`，可以把上面的 `libstdc++` 源的注释去掉。推荐使用 `clang` 的 `C++` 标准库 `libc++`，毕竟你使用的是 `clang` 而不是 `gcc`。

4) 如果要使用 `gcc` 的 `C++` 标准库 `libstdc++`，可能还需要添加一下 `ppa` 的支持：

    ```
    $ sudo add-apt-repository ppa:ubuntu-toolchain-r/test
    ```

    如果出现 `add-apt-repository` 命令未找到之类的错误提示，则执行下面命令，先安装 `add-apt-repository` 的支持：

    ```
    $ sudo apt-get install -y software-properties-common
    ```

## 2. 安装 clang ##

### 2.1 安装 old-stable branch (clang 5.0) ###

1) 获取安装包的签名

    ```
    $ wget --no-check-certificate -O - https://apt.llvm.org/llvm-snapshot.gpg.key | sudo apt-key add - # Fingerprint: 6084 F3CF 814B 57C1 CF12 EFD5 15CF 4D18 AF4F 7421
    ```

2) 执行 `update` 更新 `dep` 源

    添加了 `dep` 源和 `安装包签名` 以后，必须执行一次 “`apt-get update`” 命令后，源才能生效：

    ```
    $ sudo apt-get update
    ```

3) 如果只想安装 `clang` 和 `lldb` (5.0 release)

    ```
    $ sudo apt-get install clang-5.0 lldb-5.0
    ```

4) 如果需要安装 `clang 5.0` 所有的包

    ```
    $ sudo apt-get install clang-5.0 clang-tools-5.0 clang-5.0-doc libclang-common-5.0-dev libclang-5.0-dev libclang1-5.0 libllvm5.0 lldb-5.0 llvm-5.0 llvm-5.0-dev llvm-5.0-doc llvm-5.0-examples llvm-5.0-runtime clang-format-5.0 python-clang-5.0 libfuzzer-5.0-dev
    ```

### 2.2 安装 stable branch (clang 6.0) ###

1) 获取安装包的签名

    ```
    $ wget --no-check-certificate -O - https://apt.llvm.org/llvm-snapshot.gpg.key | sudo apt-key add - # Fingerprint: 6084 F3CF 814B 57C1 CF12 EFD5 15CF 4D18 AF4F 7421
    ```

2) 执行 `update` 更新 `dep` 源

    添加了 `dep` 源和 `安装包签名` 以后，必须执行一次 “`apt-get update`” 命令后，源才能生效：

    ```
    $ sudo apt-get update
    ```

3) 如果只想安装 `clang` 和 `lldb` (6.0 release)

    ```
    $ sudo apt-get install clang-6.0 lldb-6.0 lld-6.0
    ```

4) 如果需要安装 `clang 6.0` 所有的包

    ```
    $ sudo apt-get install clang-6.0 clang-tools-6.0 clang-6.0-doc libclang-common-6.0-dev libclang-6.0-dev libclang1-6.0 libllvm-6.0-ocaml-dev libllvm6.0 lldb-6.0 llvm-6.0 llvm-6.0-dev llvm-6.0-doc llvm-6.0-examples llvm-6.0-runtime clang-format-6.0 python-clang-6.0 lldb-6.0-dev lld-6.0 libfuzzer-6.0-dev
    ```

### 2.3 安装 development branch (clang 7.0) ###

1) 获取安装包的签名

    ```
    $ wget --no-check-certificate -O - https://apt.llvm.org/llvm-snapshot.gpg.key | sudo apt-key add - # Fingerprint: 6084 F3CF 814B 57C1 CF12 EFD5 15CF 4D18 AF4F 7421
    ```

2) 执行 `update` 更新 `dep` 源

    添加了 `dep` 源和 `安装包签名` 以后，必须执行一次 “`apt-get update`” 命令后，源才能生效：

    ```
    $ sudo apt-get update
    ```

3) 如果只想安装 `clang`、`lld` 和 `lldb` (7.0 release)

    ```
    $ sudo apt-get install apt-get install clang-7 lldb-7 lld-7
    ```

4) 如果需要安装 `clang 7.0` 所有的包

    ```
    $ sudo apt-get install clang-7 clang-tools-7 clang-7-doc libclang-common-7-dev libclang-7-dev libclang1-7 libllvm-7-ocaml-dev libllvm7 lldb-7 llvm-7 llvm-7-dev llvm-7-doc llvm-7-examples llvm-7-runtime clang-format-7 python-clang-7 lld-7 libfuzzer-7-dev
    ```

    注：`development branch` (`clang 7.0`) 的更新源好像上面好像没有给出，是否能更新笔者没验证过。
