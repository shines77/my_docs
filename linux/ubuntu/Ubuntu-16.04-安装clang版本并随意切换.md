
# Ubuntu 16.04 安装 clang 各个版本并随意切换

## 1. 添加 PPA 源

在 `toolchain/test` 下已经有打包好的 `clang` ，版本有 `3.x`、`5.0`、`6.0` 等，用这个 `PPA` 升级 `clang` 就可以啦！

首先添加 `ppa` 到更新源，并更新 `apt-get` ：

```shell
# 安装 add-apt-repository 组件
sudo apt-get install build-essential software-properties-common -y

# 添加 ppa 源
sudo add-apt-repository ppa:ubuntu-toolchain-r/test

sudo apt-get update

# 安装 gcc 快照
apt-get install gcc-snapshot

sudo apt-get update
```

其中第二句执行的时候，如果你已经安装过 `add-apt-repository` 了的话，按回车确认，继续即可。

如果出现下面的错误信息或其他错误提示，则说明 `add-apt-repository` 没有安装或无法正常工作：

```shell
sudo: add-apt-repository: command not found
```

注意：在添加了 `ppa` 源以后，一定要记得执行 `sudo apt-get update` 命令，才会让添加的 `ppa` 生效。

## 2. 安装 clang

`Ubuntu 16.04` 系统默认安装的版本是 `clang-3.8`，过于老旧，可以安装更高的版本。

如果原来已经做过 `clang` 和 `clang++` 的 `update-alternatives`，并想保留原来的设置，则该步骤可以跳过。

但如果你觉得原先的设置存在问题，可以用下面的命令清除原来的设置。

删除所有的 `/usr/bin/clang` 和 `/usr/bin/clang++` 的重定向 `link`：

```shell
sudo update-alternatives --remove-all clang
sudo update-alternatives --remove-all clang++
```

然后，分别安装各个版本的 `clang` 和 `clang++`：

```shell
sudo apt-get update

sudo apt-get install llvm

# 如果你已经使用 sudo apt-get install clang 命令
# 安装了 3.8 版本（默认版本是 3.8）
# 但是为了配置方便，可以再手动装一次 3.8 版本
sudo apt-get install clang-3.8

sudo apt-get install clang-5.0
sudo apt-get install clang-6.0
sudo apt-get install clang-7
sudo apt-get install clang-8
sudo apt-get install clang-9
sudo apt-get install clang-10
sudo apt-get install clang-11
sudo apt-get install clang-12

sudo apt-get install clang-18
```

（上次验证日期：`2020` 年 `9` 月 `12` 日。）
（最后验证日期：`2025` 年 `2` 月 `5` 日。）

**安装 clang 的 libc++ 库**

```bash
# 安装默认版本的 libc++
sudo apt-get install clang llvm libc++-dev libc++abi-dev

# 安装 clang 12.0 及对应的 libc++
sudo apt-get install clang-12 llvm-12 libc++-12-dev libc++abi-12-dev

# 安装 clang 18.0 及对应的 libc++
sudo apt-get install clang-18 llvm-18 libc++-18-dev libc++abi-18-dev
```

提示：可以使用 `apt-cache search libc++` 命令搜索。

现在要刷新一下系统数据和设置，该步骤最好做一下（推荐），否则在使用 `locate`, `which` 等命令时，是搜索不到上面更新的 `clang` 相关的文件或目录的：

```shell
sudo updatedb && sudo ldconfig

locate gcc
```

现在，你会发现 `clang --version` 显示出来的版本是前面安装命令里装过的 `clang` 的最新版本，如果我们想随意切换 `clang` 的版本，则需要更新一下软链接，我们使用 `alternatives` 来管理 `clang`，`clang++` 的软链接，方法见下一小节。

## 3. 配置 alternatives

下面是使用 `alternatives` 配置 `clang` 的软链接的命令，以便切换。

### 3.1. 添加 alternatives 配置

`clang 3.8`：

```shell
sudo update-alternatives --install /usr/bin/clang clang /usr/bin/clang-3.8 38 --slave /usr/bin/llvm-ar llvm-ar /usr/bin/llvm-ar-3.8 --slave /usr/bin/llvm-as llvm-as /usr/bin/llvm-as-3.8 --slave /usr/bin/llvm-link llvm-link /usr/bin/llvm-link-3.8 --slave /usr/bin/llvm-lib llvm-lib /usr/bin/llvm-lib-3.8 --slave /usr/bin/llvm-nm llvm-nm /usr/bin/llvm-nm-3.8 --slave /usr/bin/llvm-ranlib llvm-ranlib /usr/bin/llvm-ranlib-3.8 --slave /usr/bin/llvm-objdump llvm-objdump /usr/bin/llvm-objdump-3.8 --slave /usr/bin/llvm-cxxdump llvm-cxxdump /usr/bin/llvm-cxxdump-3.8 --slave /usr/bin/llvm-config llvm-config /usr/bin/llvm-config-3.8 --slave /usr/bin/llvm-tblgen llvm-tblgen /usr/bin/llvm-tblgen-3.8 --slave /usr/bin/clang++ clang++ /usr/bin/clang++-3.8
```

`clang 5.0`：

```shell
sudo update-alternatives --install /usr/bin/clang clang /usr/bin/clang-5.0 50 --slave /usr/bin/llvm-ar llvm-ar /usr/bin/llvm-ar-5.0 --slave /usr/bin/llvm-as llvm-as /usr/bin/llvm-as-5.0 --slave /usr/bin/llvm-link llvm-link /usr/bin/llvm-link-5.0 --slave /usr/bin/llvm-lib llvm-lib /usr/bin/llvm-lib-5.0 --slave /usr/bin/llvm-nm llvm-nm /usr/bin/llvm-nm-5.0 --slave /usr/bin/llvm-ranlib llvm-ranlib /usr/bin/llvm-ranlib-5.0 --slave /usr/bin/llvm-objdump llvm-objdump /usr/bin/llvm-objdump-5.0 --slave /usr/bin/llvm-cxxdump llvm-cxxdump /usr/bin/llvm-cxxdump-5.0 --slave /usr/bin/llvm-config llvm-config /usr/bin/llvm-config-5.0 --slave /usr/bin/llvm-tblgen llvm-tblgen /usr/bin/llvm-tblgen-5.0 --slave /usr/bin/clang++ clang++ /usr/bin/clang++-5.0
```

`clang 6.0`：

```shell
sudo update-alternatives --install /usr/bin/clang clang /usr/bin/clang-6.0 60 --slave /usr/bin/llvm-ar llvm-ar /usr/bin/llvm-ar-6.0 --slave /usr/bin/llvm-as llvm-as /usr/bin/llvm-as-6.0 --slave /usr/bin/llvm-link llvm-link /usr/bin/llvm-link-6.0 --slave /usr/bin/llvm-lib llvm-lib /usr/bin/llvm-lib-6.0 --slave /usr/bin/llvm-nm llvm-nm /usr/bin/llvm-nm-6.0 --slave /usr/bin/llvm-ranlib llvm-ranlib /usr/bin/llvm-ranlib-6.0 --slave /usr/bin/llvm-objdump llvm-objdump /usr/bin/llvm-objdump-6.0 --slave /usr/bin/llvm-cxxdump llvm-cxxdump /usr/bin/llvm-cxxdump-6.0 --slave /usr/bin/llvm-config llvm-config /usr/bin/llvm-config-6.0 --slave /usr/bin/llvm-tblgen llvm-tblgen /usr/bin/llvm-tblgen-6.0 --slave /usr/bin/clang++ clang++ /usr/bin/clang++-6.0
```

`clang 7.0`：

```shell
sudo update-alternatives --install /usr/bin/clang clang /usr/lib/llvm-7/bin/clang 70 --slave /usr/bin/llvm-ar llvm-ar /usr/bin/llvm-ar-7 --slave /usr/bin/llvm-as llvm-as /usr/bin/llvm-as-7 --slave /usr/bin/llvm-link llvm-link /usr/bin/llvm-link-7 --slave /usr/bin/llvm-lib llvm-lib /usr/bin/llvm-lib-7 --slave /usr/bin/llvm-nm llvm-nm /usr/bin/llvm-nm-7 --slave /usr/bin/llvm-ranlib llvm-ranlib /usr/bin/llvm-ranlib-7 --slave /usr/bin/llvm-objdump llvm-objdump /usr/bin/llvm-objdump-7 --slave /usr/bin/llvm-cxxdump llvm-cxxdump /usr/bin/llvm-cxxdump-7 --slave /usr/bin/llvm-config llvm-config /usr/bin/llvm-config-7 --slave /usr/bin/llvm-tblgen llvm-tblgen /usr/bin/llvm-tblgen-7 --slave /usr/bin/clang++ clang++ /usr/lib/llvm-7/bin/clang++
```

`clang 8.0`：

```shell
sudo update-alternatives --install /usr/bin/clang clang /usr/bin/clang-8 80 --slave /usr/bin/llvm-ar llvm-ar /usr/bin/llvm-ar-8 --slave /usr/bin/llvm-as llvm-as /usr/bin/llvm-as-8 --slave /usr/bin/llvm-link llvm-link /usr/bin/llvm-link-8 --slave /usr/bin/llvm-lib llvm-lib /usr/bin/llvm-lib-8 --slave /usr/bin/llvm-nm llvm-nm /usr/bin/llvm-nm-8 --slave /usr/bin/llvm-ranlib llvm-ranlib /usr/bin/llvm-ranlib-8 --slave /usr/bin/llvm-objdump llvm-objdump /usr/bin/llvm-objdump-8 --slave /usr/bin/llvm-cxxdump llvm-cxxdump /usr/bin/llvm-cxxdump-8 --slave /usr/bin/llvm-config llvm-config /usr/bin/llvm-config-8 --slave /usr/bin/llvm-tblgen llvm-tblgen /usr/bin/llvm-tblgen-8 --slave /usr/bin/clang++ clang++ /usr/bin/clang++-8
```

`clang 9.0`：

```shell
sudo update-alternatives --install /usr/bin/clang clang /usr/bin/clang-9 90 --slave /usr/bin/llvm-ar llvm-ar /usr/bin/llvm-ar-9 --slave /usr/bin/llvm-as llvm-as /usr/bin/llvm-as-9 --slave /usr/bin/llvm-link llvm-link /usr/bin/llvm-link-9 --slave /usr/bin/llvm-lib llvm-lib /usr/bin/llvm-lib-9 --slave /usr/bin/llvm-nm llvm-nm /usr/bin/llvm-nm-9 --slave /usr/bin/llvm-ranlib llvm-ranlib /usr/bin/llvm-ranlib-9 --slave /usr/bin/llvm-objdump llvm-objdump /usr/bin/llvm-objdump-9 --slave /usr/bin/llvm-cxxdump llvm-cxxdump /usr/bin/llvm-cxxdump-9 --slave /usr/bin/llvm-config llvm-config /usr/bin/llvm-config-9 --slave /usr/bin/llvm-tblgen llvm-tblgen /usr/bin/llvm-tblgen-9 --slave /usr/bin/clang++ clang++ /usr/bin/clang++-9
```

`clang 10.0`：

```shell
sudo update-alternatives --install /usr/bin/clang clang /usr/bin/clang-10 100 --slave /usr/bin/llvm-ar llvm-ar /usr/bin/llvm-ar-10 --slave /usr/bin/llvm-as llvm-as /usr/bin/llvm-as-10 --slave /usr/bin/llvm-link llvm-link /usr/bin/llvm-link-10 --slave /usr/bin/llvm-lib llvm-lib /usr/bin/llvm-lib-10 --slave /usr/bin/llvm-nm llvm-nm /usr/bin/llvm-nm-10 --slave /usr/bin/llvm-ranlib llvm-ranlib /usr/bin/llvm-ranlib-10 --slave /usr/bin/llvm-objdump llvm-objdump /usr/bin/llvm-objdump-10 --slave /usr/bin/llvm-cxxdump llvm-cxxdump /usr/bin/llvm-cxxdump-10 --slave /usr/bin/llvm-config llvm-config /usr/bin/llvm-config-10 --slave /usr/bin/llvm-tblgen llvm-tblgen /usr/bin/llvm-tblgen-10 --slave /usr/bin/clang++ clang++ /usr/bin/clang++-10
```

`clang 11.0`：

```shell
sudo update-alternatives --install /usr/bin/clang clang /usr/bin/clang-11 110 --slave /usr/bin/llvm-ar llvm-ar /usr/bin/llvm-ar-11 --slave /usr/bin/llvm-as llvm-as /usr/bin/llvm-as-11 --slave /usr/bin/llvm-link llvm-link /usr/bin/llvm-link-11 --slave /usr/bin/llvm-lib llvm-lib /usr/bin/llvm-lib-11 --slave /usr/bin/llvm-nm llvm-nm /usr/bin/llvm-nm-11 --slave /usr/bin/llvm-ranlib llvm-ranlib /usr/bin/llvm-ranlib-11 --slave /usr/bin/llvm-objdump llvm-objdump /usr/bin/llvm-objdump-11 --slave /usr/bin/llvm-cxxdump llvm-cxxdump /usr/bin/llvm-cxxdump-11 --slave /usr/bin/llvm-config llvm-config /usr/bin/llvm-config-11 --slave /usr/bin/llvm-tblgen llvm-tblgen /usr/bin/llvm-tblgen-11 --slave /usr/bin/clang++ clang++ /usr/bin/clang++-11
```

`clang 12.0`：

```shell
sudo update-alternatives --install /usr/bin/clang clang /usr/bin/clang-12 120 --slave /usr/bin/llvm-ar llvm-ar /usr/bin/llvm-ar-12 --slave /usr/bin/llvm-as llvm-as /usr/bin/llvm-as-12 --slave /usr/bin/llvm-link llvm-link /usr/bin/llvm-link-12 --slave /usr/bin/llvm-lib llvm-lib /usr/bin/llvm-lib-12 --slave /usr/bin/llvm-nm llvm-nm /usr/bin/llvm-nm-12 --slave /usr/bin/llvm-ranlib llvm-ranlib /usr/bin/llvm-ranlib-12 --slave /usr/bin/llvm-objdump llvm-objdump /usr/bin/llvm-objdump-12 --slave /usr/bin/llvm-cxxdump llvm-cxxdump /usr/bin/llvm-cxxdump-12 --slave /usr/bin/llvm-config llvm-config /usr/bin/llvm-config-12 --slave /usr/bin/llvm-tblgen llvm-tblgen /usr/bin/llvm-tblgen-12 --slave /usr/bin/clang++ clang++ /usr/bin/clang++-12
```

`clang 18.1.8`：

```shell
sudo update-alternatives --install /usr/bin/clang clang /usr/bin/clang-18 181 --slave /usr/bin/llvm-ar llvm-ar /usr/bin/llvm-ar-18 --slave /usr/bin/llvm-as llvm-as /usr/bin/llvm-as-18 --slave /usr/bin/llvm-link llvm-link /usr/bin/llvm-link-18 --slave /usr/bin/llvm-lib llvm-lib /usr/bin/llvm-lib-18 --slave /usr/bin/llvm-nm llvm-nm /usr/bin/llvm-nm-18 --slave /usr/bin/llvm-ranlib llvm-ranlib /usr/bin/llvm-ranlib-18 --slave /usr/bin/llvm-objdump llvm-objdump /usr/bin/llvm-objdump-18 --slave /usr/bin/llvm-cxxdump llvm-cxxdump /usr/bin/llvm-cxxdump-18 --slave /usr/bin/llvm-config llvm-config /usr/bin/llvm-config-18 --slave /usr/bin/llvm-tblgen llvm-tblgen /usr/bin/llvm-tblgen-18 --slave /usr/bin/clang++ clang++ /usr/bin/clang++-18
```

## 4. 切换 clang 版本

### 4.1. 切换 `clang` 版本的命令

```shell
sudo update-alternatives --config clang
```

例如，有 4 个候选项可用于替换 `clang` （默认路径 `/usr/bin/clang`），如下所示：

```shell
  选择       路径            优先级  状态
------------------------------------------------------------
* 0            /usr/bin/clang-3.8   38        自动模式
  1            /usr/bin/clang-5.0   50        手动模式
  2            /usr/bin/clang-6.0   60        手动模式
  3            /usr/bin/clang-8     80        手动模式
  4            /usr/bin/clang-9     90        手动模式
  5            /usr/bin/clang-10   100        手动模式
  6            /usr/bin/clang-11   110        手动模式
  7            /usr/bin/clang-12   120        手动模式
```

注：为了方便，切换 `clang` 的同时，我们也切换到相同版本的 `clang++`。

如果想单独切换 `clang++` 的版本，可以参考 `clang` 的版本，修改上一小节的 `update-alternatives` 配置。

同理，`alternatives` 除了可以管理 `clang`，`clang++` 以外，任何其他系列工具软件都可以使用 `alternatives` 来管理和切换，比如：`gcc`。

## 5. 参考文章

----------------------------------------------------------------

* [http://www.cnblogs.com/BlackStorm/p/5183490.html](http://www.cnblogs.com/BlackStorm/p/5183490.html)

* [http://blog.sina.com.cn/s/blog_54dd80920102vvt6.html](http://www.cnblogs.com/BlackStorm/p/5183490.html)

----------------------------------------------------------------

## 6. 更新历史：

* `2025` / `02` / `05` ：增加 `clang 18.1`, 版本 (Ubuntu 20.04)。
* `2022` / `03` / `20` ：增加 `clang 10.x`, `clang 11.x`, `clang 12.x` 版本 (Ubuntu 20.04)。
* `2022` / `03` / `18` ：增加 `clang 9.x` 版本 (Ubuntu 16.04)。
* `2020` / `09` / `12` ：`clang` 的第一个版本。

<.end.>
