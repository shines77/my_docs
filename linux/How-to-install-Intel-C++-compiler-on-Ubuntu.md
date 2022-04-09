# 如何在 Ubuntu 上安装 Intel C++ Compiler

## 1. 使用 oneapi 安装源

首先，添加 Intel 的 `oneapi` 安装源：

```shell
echo "deb https://apt.repos.intel.com/oneapi all main" >/etc/apt/sources.list.d/intel-oneapi.list
apt-get update
apt-get install intel-oneapi-compiler-dpcpp-cpp-and-cpp-classic
```

但是当 update 的时候，得到如下信息：

```text
Get:5 https://apt.repos.intel.com/oneapi all InRelease [5,673 B]                 
Err:5 https://apt.repos.intel.com/oneapi all InRelease
  The following signatures couldn't be verified because the public key is not available: NO_PUBKEY ACFA9FC57E6C5DBE
Reading package lists... Done
W: GPG error: https://apt.repos.intel.com/oneapi all InRelease: The following signatures couldn't be verified because the public key is not available: NO_PUBKEY ACFA9FC57E6C5DBE
E: The repository 'https://apt.repos.intel.com/oneapi all InRelease' is not signed.
N: Updating from such a repository can't be done securely, and is therefore disabled by default.
N: See apt-secure(8) manpage for repository creation and user configuration details
```

大意是缺少密钥： `ACFA9FC57E6C5DBE`。

`Google` 之后，得到如下信息：

```text
The apt packaging system has a set of trusted keys that determine whether a package can be authenticated and therefore trusted to be installed on the system. Sometimes the system does not have all the keys it needs and runs into this issue. Fortunately, there is a quick fix. Each key that is listed as missing needs to be added to the apt key manager so that it can authenticate the packages.
```

译文：`apt` 打包系统有一组受信任的密钥，用于确定一个安装包是否可以通过身份验证，如果是受信任的包，才安装到系统上。有时系统不一定拥有所有你所需的密钥，因此会遇到此问题。幸运的是，有一个快速解决方案。把每一个缺失的密钥，都添加到 `apt` 密钥管理器中，以便它能够对安装包进行身份验证。

解决方法：

```shell
sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys ACFA9FC57E6C5DBE

# 返回结果
Executing: /tmp/apt-key-gpghome.C1ToTyJD6A/gpg.1.sh --keyserver keyserver.ubuntu.com --recv-keys ACFA9FC57E6C5DBE
gpg: key ACFA9FC57E6C5DBE: 3 signatures not checked due to missing keys
gpg: key ACFA9FC57E6C5DBE: public key "Intel(R) Software Development Products" imported
gpg: Total number processed: 1
gpg:               imported: 1
```

再次执行 "`apt-get update`"，这次就 OK 了。安装文件有 `3741` MB，有点大。。

安装完成以后，执行下面命令：

```shell
$ source /opt/intel/oneapi/setvars.sh

:: initializing oneAPI environment ...
   bash: BASH_VERSION = 4.4.20(1)-release
   args: Using "$@" for setvars.sh arguments: 
:: compiler -- latest
:: debugger -- latest
:: dev-utilities -- latest
:: tbb -- latest
:: oneAPI environment initialized ::

$ which icpc
/opt/intel/oneapi/compiler/2022.0.2/linux/bin/intel64/icpc

$ icpc -V
icpc version 2021.5.0 (gcc version 9.4.0 compatibility)
```

有一点需要注意的是，要使用 "`icc`" 或 "`icpc`"，你必须每次都终端里执行一遍 "`source /opt/intel/oneapi/setvars.sh`"，才能使用，否则只能使用完整的路径启动 "`icc`" 或 "`icpc`"。

## 2. Intel 官网在线安装

### 2.1 在线安装

先打开 Intel 官网的下载页面：[Intel® oneAPI DPC++/C++ Compiler and Intel® C++ Compiler Classic](https://www.intel.com/content/www/us/en/developer/articles/tool/oneapi-standalone-components.html#dpcpp-cpp) 。

下载信息如下：

[**2022.0.2 Release**]

| Name (Click to initiate download)    | Version | Size | Installer | Date |
|:-------------------------------------|:--------|:-----|:----------|:-----|
|Intel® DPC++/C++ Compiler for Linux*  | 2022.0.2 | 16.6 MB  | Online   |Jan. 21, 2022|
|Intel DPC++/C++ Compiler for Linux    | 2022.0.2 | 1.14 GB  | Offline  |Jan. 21, 2022|
|Intel DPC++/C++ Compiler for Windows* | 2022.0.3 | 14 MB    | Online   |Mar. 03, 2022|
|Intel DPC++/C++ Compiler for Windows  | 2022.0.3 | 1.06 GB  | Offline  |Mar. 03, 2022|
|Intel C++ Compiler Classic for macOS* | 2022.0.0 | 23.55 MB | Online   |Dec. 02, 2021|
|Intel C++ Compiler Classic for macOS  | 2022.0.0 | 93.7 MB  | Offline  |Dec. 02, 2021|

选择 `Linux` 的 online 版本：[Intel® DPC++/C++ Compiler for Linux*: Online](https://registrationcenter-download.intel.com/akdlm/irc_nas/18478/l_dpcpp-cpp-compiler_p_2022.0.2.84.sh)

下载并运行安装脚本：

```shell
wget -c https://registrationcenter-download.intel.com/akdlm/irc_nas/18478/l_dpcpp-cpp-compiler_p_2022.0.2.84.sh

chmod +x l_dpcpp-cpp-compiler_p_2022.0.2.84.sh

./l_dpcpp-cpp-compiler_p_2022.0.2.84.sh
```

运行安装脚本后，会看到如下欢迎信息：

```text
  Welcome to Intel® Software Installer
| Intel® oneAPI DPC++/C++ Compiler & Intel® C++ Compiler Classic
--------------------------------------------------------------------------------
  Standards driven high performance cross architecture compiler and high
  performance C++ CPU focused compiler

  Check the default configuration below.

  It can be customized before installing or downloading.

  WHAT'S INCLUDED:

    - Intel® oneAPI DPC++/C++ Compiler & Intel® C++ Compiler Classic (2022.0.2
    )

  INSTALLATION LOCATION: /opt/intel/oneapi

  Intel® Software Installer: *4.1.0.0-101SPACE REQUIRED TO DOWNLOAD: 1.1 GB
                              
  By continuing with this installation, you accept the terms and conditions of
  Intel® End User License Agreement

  | Accept & install | Accept & customize installation | Download Only | Decline & quit |
```

选择 "`Accept & install`"，中途选择 "`Skip Eclipse* IDE Configuration`"，其他选择自行决定。最后，开始下载和安装。另外一点，在线安装的版本只有 1.1 GB（可能是指下载的文件大小，不是安装后的大小）。

### 2.2 离线安装

离线版的安装方法和 `Online` 版类似，下载页面跟 "`2.1 在线安装`" 小节的一样。

下载离线安装脚本：

```shell
wget -c https://registrationcenter-download.intel.com/akdlm/irc_nas/18478/l_dpcpp-cpp-compiler_p_2022.0.2.84_offline.sh

chmod +x l_dpcpp-cpp-compiler_p_2022.0.2.84_offline.sh

./l_dpcpp-cpp-compiler_p_2022.0.2.84_offline.sh
```

其他应该跟 `Online` 版是类似的，不再敖述（我自己也没有实装过，应该是相似的）。

## 3. 参考文章

- [Intel® oneAPI DPC++/C++ Compiler and Intel® C++ Compiler Classic](https://www.intel.com/content/www/us/en/developer/articles/tool/oneapi-standalone-components.html#dpcpp-cpp) [www.intel.com]

- [Intel C++ compiler is free (as in beer) as part of oneAPI](https://www.reddit.com/r/cpp/comments/kafmsz/intel_c_compiler_is_free_as_in_beer_as_part_of/) [www.reddit.com]

- [Fix apt-get update “the following signatures couldn’t be verified because the public key is not available”](https://chrisjean.com/fix-apt-get-update-the-following-signatures-couldnt-be-verified-because-the-public-key-is-not-available/) [chrisjean.com]
