# Linux 下 GDB 调试命令

## 1. 调试

基础要求：要调试应用程序，`gcc` 的编译选项必须加入 `'-O0 -g'`，至少得使用 `'-g'`。

命令帮助：敲入 `b` 按两次 `TAB` 键，你会看到所有 `b` 打头的命令：

```shell
(gdb) b
backtrace  break      bt
(gdb)
```

### 1.1 GDB 命令

#### 1.1.1 调试已运行的程序

两种方法：

1. 在 `UNIX` 下用 `ps` 查看正在运行的程序的 `PID (进程 ID)`，然后用 `gdb <program> PID` 格式挂接正在运行的程序。

2. 先用 `gdb <program>` 关联上源代码，并进行 `gdb` ，在 `gdb` 中用 `attach` 命令来挂接进程的 `PID` 。并用 `detach` 来取消挂接的进程。

#### 1.1.2 暂停/恢复程序运行

调试程序中，暂停程序运行是必须的， `gdb` 可以方便地暂停程序的运行。你可以设置程序的在哪行停住，在什么条件下停住，在收到什么信号时停往等等。以便于你查看运行时的变量，以及运行时的流程。

当进程被 `gdb` 停住时，你可以使用 `info program` 来查看程序的是否在运行，进程号，被暂停的原因。

在 `gdb` 中，我们可以有以下几种暂停方式：断点（`BreakPoint`）、观察点（`WatchPoint`）、捕捉点（`CatchPoint`）、信号（`Signals`）、线程停止（`Thread Stops`）。如果要恢复程序运行，可以使用 `c` 或是 `continue` 命令。

#### 1.1.3 设置断点（BreakPoint）

* 设置断点

  1. break \<function\>

        b / break \<function\>，例如：b main，把断点设置在 main() 函数上。

  2. break \<linenum\>

        在指定行号停住。

  3. break \<filename:linenum\>

        在源文件 filename 的 linenum 行处停住。

  4. break \<filename:function\>

        在源文件 filename 的 function 函数的入口处停住。

  5. break *address

        在程序运行的内存地址处停住。

  6. break

        break 命令没有参数时，表示在下一条指令处停住

恢复程序运行和单步调试

* 单步跟踪：s / step \<count\>，例如：s 10。

    单步跟踪，如果有函数调用，他会进入该函数。

* 单步跟踪：n / next \<count\>，例如：n 10。

    同样单步跟踪，如果有函数调用，他不会进入该函数，很像 VC 等工具中的 step over 。

* 设置 step-mode

  * set step-mode on

    打开 step-mode 模式，于是，在进行单步跟踪时，程序不会因为没有 debug 信息而不停住。这个参数有很利于查看机器码。

  * set step-mod off

    关闭 step-mode 模式。

当程序被停住了，你可以用 continue 命令恢复程序的运行直到程序结束，或下一个断点到来。也可以使用 step 或 next 命令单步跟踪程序。

* continue [ignore-count]
* c [ignore-count]
* fg [ignore-count]

    恢复程序运行，直到程序结束，或是下一个断点到来。 ignore-count 表示忽略其后的断点次数。 continue，c，fg 三个命令都是一样的意思。

#### 1.1.4 设置观察点（WatchPoint）

观察点一般来观察某个表达式（变量也是一种表达式）的值是否有变化了，如果有变化，马上停住程序。我们有下面的几种方法来设置观察点：

* watch \<expr\>

    为表达式（变量）expr 设置一个观察点。一量表达式值有变化时，马上停住程序。

* rwatch \<expr\>

    当表达式（变量）expr 被读时，停住程序。

* awatch \<expr\>

    当表达式（变量）的值被读或被写时，停住程序。

* info watchpoints

    列出当前所设置了的所有观察点。

#### 1.1.5 设置捕捉点（CatchPoint）

你可设置捕捉点来补捉程序运行时的一些事件。如：载入共享库（动态链接库）或是 C++ 的异常。设置捕捉点的格式为：

* catch \<event\>

   当 event 发生时，停住程序。event 可以是下面的内容：

   1. throw 一个 C++ 抛出的异常。（ throw 为关键字）
   2. catch 一个 C++ 捕捉到的异常。（ catch 为关键字）
   3. exec 调用系统调用 exec 时。（ exec 为关键字，目前此功能只在 HP-UX 下有用）
   4. fork 调用系统调用 fork 时。（ fork 为关键字，目前此功能只在 HP-UX 下有用）
   5. vfork 调用系统调用 vfork 时。（ vfork 为关键字，目前此功能只在 HP-UX 下有用）
   6. load 或 `load <libname>` 载入共享库（动态链接库）时。（load 为关键字，目前此功能只在 HP-UX 下有用）
   7. unload 或 `unload <libname>` 卸载共享库（动态链接库）时。（unload 为关键字，目前此功能只在 HP-UX 下有用）

* tcatch \<event\>

   只设置一次捕捉点，当程序停住以后，应点被自动删除。

#### 1.1.6 维护停止点

上面说了如何设置程序的停止点，GDB 中的停止点也就是上述的三类。在 GDB 中，如果你觉得已定义好的停止点没有用了，你可以使用 delete、clear、disable、enable 这几个命令来进行维护。

* clear

    清除所有的已定义的停止点。

* clear \<function\>
* clear \<filename:function\>

    清除所有设置在函数上的停止点。

* clear \<linenum\>
* clear \<filename:linenum\>

    清除所有设置在指定行上的停止点。

* delete [breakpoints] [range...]

    删除指定的断点， breakpoints 为断点号。如果不指定断点号，则表示删除所有的断点。 range 表示断点号的范围（如： 3-7 ）。其简写命令为 d 。

比删除更好的一种方法是 disable 停止点，disable 了的停止点， GDB 不会删除，当你还需要时， enable 即可，就好像回收站一样。

* disable [breakpoints] [range...]

    disable 所指定的停止点， breakpoints 为停止点号。如果什么都不指定，表示 disable 所有的停止点。简写命令是 dis.

* enable [breakpoints] [range...]

    enable 所指定的停止点， breakpoints 为停止点号。

* enable [breakpoints] once range...

    enable 所指定的停止点一次，当程序停止后，该停止点马上被 GDB 自动 disable 。

* enable [breakpoints] delete range...

    enable 所指定的停止点一次，当程序停止后，该停止点马上被 GDB 自动删除。

## 2. GDB 调试汇编

* 能够实时看到寄存器值的变化

使用 `gdb` 时增加 `-tui` 选项，打开 `gdb` 后运行 `layout regs` 命令。注意最好加上 `-tui`，否则很大可能会出现花屏现象。

* 能够看到源代码和对应汇编的关系

在 `gdb` 中运行 `set disassemble-next-line on`，表示自动反汇编后面要执行的代码。

## 3. 参考文章

* `Linux 下 gdb 单步调试`

    [https://blog.csdn.net/zzymusic/article/details/4815142]()

* `GDB 单步调试汇编`

    [https://www.cnblogs.com/zhangyachen/p/9227037.html]()
