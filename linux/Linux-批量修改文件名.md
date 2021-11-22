# Linux 下批量修改文件名

## 1. 批量修改文件名

`Linux` 下修改文件名有 `mv` 和 `rename`。其中 `mv` 命令只能对单个文件重命名，这是 `mv` 命令和 `rename` 命令的在重命名方面的根本区别。另外，还可以使用 `mmv` 命令，这个命令更像 `Windows` 的 `Dos` 下的 `ren` 命令。

### 1.1 方法一 rename

`Linux` 的 `rename` 命令有两个版本，一个是 `c` 语言版本的，一个是 `perl` 语言版本的，判断方法：

输入 `man rename` 看到第一行是如下信息，那就是 `C` 语言版本的：

```shell
RENAME(1) Linux Programmer’s Manual RENAME(1)
```

而如果出现如下信息，则是 `Perl` 版本：

```shell
RENAME(1) Perl Programmers Reference Guide RENAME(1)
```

注: 要使用 `man` 手册, 可以先用如下命令安装手册（英文版和中文版）：

```shell
sudo apt-get install manpages
sudo apt-get install manpages-zh
```

有些 `Linux` 版本的使用手册可能命令的信息不存在，例如：`Ubuntu 18.04`。

**C 语言版本格式**：

```shell
rename 原字符串 新字符串 文件名
```

**Perl 语言版本格式**：

```shell
rename 's/原字符串/新字符串/' 文件名
```

**rename 正则表达式的范例**

例如：

```perl
字符替换
rename "s/AA/aa/" *         // 把文件名中的 AA 替换成 aa

修改文件名的后缀
rename "s//.html//.php/" *  // 把 .html 后缀的改成 .php 后缀

批量添加文件名的后缀
rename "s/$//.txt/" *       // 把所有的文件名都以 txt 结尾

批量删除文件名的后缀
rename "s//.txt//" *        // 把所有以 .txt 结尾的文件名的 .txt 删掉
```

### 1.2 方法二 mv

使用 `find` 命令，找到包含 "_" 的文件夹，然后使用 `mv` 命令逐个替换：

```shell
find ./ -name "*_*" | while read id; do mv $id ${id/_/-}; done
```

以上执行后的效果是，把当前目录下的所有文件的 "`_`" 替换成 "`_`"。

### 1.3 方法三 mmv

`mmv` 命令的用法跟 `Dos` 的 `ren` 命令很像，但更强大，支持类似 `Dos` 的 `ren` 命令的通配符 `"*"`，`"?"` 等。

#### 1.3.1 安装 `mmv` 命令

`CentOS`：

在默认情况下，`CentOS 7` 的网络源中没有 `mmv` 的安装包，我们需要先安装 `epel` 源，然后再安装 `mmv` 工具。

```shell
# 下载阿里云的 epel 源文件。
wget http://mirrors.aliyun.com/repo/epel-7.repo --directory-prefix=/etc/yum.repos.d

# 清除 yum 缓存，并重新生成缓存
yum clean all && yum makecache

# 安装mmv
yum -y install mmv
```

`Ubuntu`：

（仅在 `Ubuntu 18.04` 版本上验证，其他版本请自行测试）

```shell
apt install mmv
```

#### 1.3.2 范例

1) `file01.rar.zip.tgz` 替换成 `file01.rar`：

    ```shell
    mmv '*.*.*.*' '#1.#2'
    ```

    效果：

    ```text
    file01.rar.zip.tgz
    file02.rar.zip.tgz
    file03.rar.zip.tgz
    file04.rar.zip.tgz
    ```

    批量替换成：

    ```text
    file01.rar
    file02.rar
    file03.rar
    file04.rar
    ```

2) `file01.rar` 替换成 `text01.zip`：

    ```shell
    mmv 'file*.rar' 'text#1.zip'
    ```

    效果：

    ```text
    file01.rar
    file02.rar
    file03.rar
    file04.rar
    ```

    批量替换成：

    ```text
    text01.zip
    text02.zip
    text03.zip
    text04.zip
    ```

3) `IMG01.jpeg` 替换成 `IMG01.jpg`：

    ```shell
    mmv '*.jpeg' '#1.jpg'
    ```

    效果：

    ```text
    IMG00.jpeg
    IMG01.jpeg
    IMG02.jpeg
    IMG03.jpeg
    ```

    批量替换成：

    ```text
    IMG00.jpg
    IMG01.jpg
    IMG02.jpg
    IMG03.jpg
    ```

4) 把 `index.html.cn` 替换成 `index.cn.html`：

    ```shell
    mmv '*.html.??' '#1.#2#3.html'
    ```

    注：其中的两个 `"?"` 通配符分别代表 `"#2"`，`"#3"` 两个占位符。

    效果：

    ```text
    index.html.cn
    index.html.en
    index.html.de
    index.html.fr
    ```

    批量替换成：

    ```text
    index.cn.html
    index.en.html
    index.de.html
    index.fr.html
    ```

## 2. 参考文章

* [Linux 下批量修改文件名](https://www.jianshu.com/p/bdd27936416e)

* [使用 mmv 命令批量修改文件名称](https://www.linuxprobe.com/linux-mmv-rename.html)
