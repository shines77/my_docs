# Linux shell 登录时启动脚本的执行顺序

## 1. 起因

因为在 `Ubuntu 20.04` 上安装了 `WebAssembly` 的 [emscripten](https://emscripten.org/)，导致了 `ll` 命令不能使用了，很不方便。

`ll` 命令其实是 "`ls -alF`" 的 `alias` （别名），并不是一个真实的命令。它是写在 `~/.bashrc` 文件里的，也就是说 `~/.bashrc` 没有被执行。

```bash
vim ~/.bashrc

# some more ls aliases
alias ll='ls -alF'
alias la='ls -A'
alias l='ls -CF'
```

`emscripten` 的安装脚本 [emsdk](https://emscripten.org/docs/getting_started/downloads.html)，在 `~/.bash_profile` 文件里写了一条命令（只有下面这一条语句），如下：

```bash
source "/home/git/emsdk/emsdk_env.sh"
```

开始我也没太留意，看不出啥毛病，但是对比了另一台 `ll` 命令正常的 `Ubuntu` 服务器，发现上面并没有这个文件。所以，我尝试着把这一条命令移至 `/etc/profile` 文件中，并删除了 `~/.bash_profile` 文件，重启，`ll` 命令就正常了。

综上所得，问题的根本原因是，如果 `~/.bash_profile` 文件存在，那么就不会再执行 `~/.profile`，也就不会执行 `~/.bashrc`，如果非要使用 `~/.bash_profile`，必须模仿 `~/.profile` ，在里面再调用 `~/.bashrc` 。

```bash
if [ "$BASH" ]; then
  if [ -f ~/.bashrc ]; then
    . ~/.bashrc
  fi
fi
```

所以是时候写篇文章，好好研究一下这些 `login shell` 自启脚本的执行顺序，因为我发现很少有文章提到这些细节。

## 2. shell 登录类型

按登录类型分：

* `login shell`：

    使用 `–login` 参数登录 `bash`，就称为 `login shell`。比如使用 `tty1` - `tty6` 登录，或者一般的 `SSH` 终端登录，需要输入用户的账号与密码，此时取得的 `bash` 就称为 “`login shell`”。

    `login shell` 读取的文件和执行顺序，请看下一小节：`3. 执行顺序` 。

* `non-login shell`：

    使用 `–nologin` 参数登录 `bash`，不需要重复输入用户名和密码的，称为 `non-login shell`。比如，我们登陆 `Linux` 后， 启动终端 `Terminal`，此时那个终端接口并不需要再次输入账号和密码，那个 `bas`h 的环境就称为 `non-login shell` 。又或者，你在原本的 `bash` 环境下再次调用 `bash` 命令，建立了一个 `bash` 子进程，同样的也没有输入账号和密码， 那第二个 `bash` (子进程) 也是 `non-login shell`。

    `non-login shell` 只会读取 `~/.bashrc` 这一个文件，并会从其父进程处继承环境变量。

按交互方式分：

* `interactive shell`：

    交互式 `shell`，用户默认登录之后就是交互式 `shell`。

* `non-interactive shell`：

    非交互式 `shell`，当运行 `shell` 脚本时，`bash` 是非交互式的。

## 3. 执行顺序

下面我们来研究一下 "`login shell`" 的脚本执行顺序。

我们分别在所有可能的执行脚本里，头尾加上如下的代码：

```bash
echo "/etc/profile enter."

......

echo "/etc/profile over."
```

可能的执行脚本：

```shell
/etc/profile
/etc/rc.local
~/.bash_profile
~/.bash_login
~/.profile
~/.bashrc
```

重新登录 `SSH`，观察登录后显示的信息。

得到脚本执行的顺序，如下：

1. `/etc/profile` ：

    * 如果文件 `/etc/bash.bashrc` 存在的话，则运行；
    * 如果目录 `/etc/profile.d` 存在，则运行 `/etc/profile.d` 目录下面的所有 `.sh` 脚本。

2. 如果文件 `~/.bash_profile` 不存在，则跳到第 (3) 步。如果存在，则运行该文件，且不会再执行后续的脚本。

    * 你可以在这里加入自己的代码，并模仿 `~/.profile` 调用 `~/.bashrc`；

3. 如果文件 `~/.bash_login` 不存在，则跳到第 (4) 步。如果存在，则运行该文件，且不会再执行后续的脚本。

    * 你可以在这里加入自己的代码；

4. 如果文件 `~/.profile` 存在，则运行该文件。

    * 运行 `~/.bashrc`，这里面执行 `ll` 命令的别名命令；

注：`~/.bash_profile`，`~/.bash_login` 和 `~/.profile` 三个文件只会执行其中某一个，执行顺序就是以上的顺序。其中，`/etc/rc.local` 并没有被执行。

当 "`login shell`" 退出登录的时候，会执行 `~/.bash_logout`，如果这个文件存在的话。

## 4. 后记

写完以后，我百度了一下，还是能搜到一些关于 `shell` 登录时启动脚本的执行顺序的文章的，请看下面第 4 节的 “参考文章”。里面提到的都是 `~/.bash_profile`，而没有 `~/.profile`，我使用的是 `Ubuntu 20.04`，有可能已经废弃了 `~/.bash_profile`，而改用 `~/.profile`，看下面的文章知道，两者作用是一模一样的，只是更换了名字。`emsdk` 可能并没有考虑到 `Ubuntu 20.04` 的这种改变，所以造成了文章一开始遇到的问题。

另外，`Ubuntu 20.04` 也没有 `/etc/bashrc` 文件，而是 `/etc/bash.bashrc` 。

本来写到这里，就结束了。但想了一下，好像对 `login shell`，`non-login shell` 讲得太模糊了，我就再百度了一下，也就是参考文章的 (3)、(4)，于是新增了第 2 小节：`shell 登录类型`，并新增了 `~/.bash_login` 和 `~/.bash_logout` 脚本，更完整了。

## 5. 参考文章

1. `[用户登录shell时的脚本顺序]`

    [https://blog.51cto.com/hanksole/1773501](https://blog.51cto.com/hanksole/1773501)

2. `[百度文库：登录shell时执行的启动脚本和顺序]`

    [https://wenku.baidu.com/view/726b4e36c6da50e2524de518964bcf84b9d52d34.html](https://wenku.baidu.com/view/726b4e36c6da50e2524de518964bcf84b9d52d34.html)

3. `[linux系统用户登录时脚本执行顺序]`

    [http://www.javashuo.com/article/p-sdlnyfwy-dy.html](http://www.javashuo.com/article/p-sdlnyfwy-dy.html)

4. `[linux中的login shell和non-login shell重点解析]`

    [https://blog.csdn.net/lws123253/article/details/89315218](https://blog.csdn.net/lws123253/article/details/89315218)
