# Linux 下 GDB 调试命令

## 1. GDB 调试

### 1.1 基本要求

要基于源码调试程序，`gcc` 的编译选项至少要加入 `'-g'`。某些时候，为了便于调试，可把优化级别调低，使用 `'-O0 -g'` 。

### 1.2 帮助命令

在 `gdb` 状态，可以使用 `help <cmd>` 查询 `gdb` 命令的使用帮助，例如：

```shell
(gdb) help breakpoints   # 显示所有断点 (break) 相关的命令
```

同时，也可以像 `Linux` 命令行一样，敲入命令时，可使用 "`Tab`" 键查询所有可能的命令。

例如，敲入 `b`，再按两次 "`Tab`" 键，你会看到所有 `b` 打头的命令：

```shell
(gdb) b
backtrace  break      bt
(gdb)
```

## 2 GDB 命令

### 2.1.1 调试已运行的程序

两种方法：

1. 在 `UNIX` 下用 `ps` 查看正在运行的程序的 `PID (进程 ID)`，然后用 `gdb <your_exec_file> PID` 格式挂接正在运行的程序。

    ```shell
    $ gdb ./test_exec 6022   # 关联要调试的程序，并挂接到进程 PID
    ```

2. 先用 `gdb <your_exec_file>` 关联要调试的运行的程序（源代码），然后在 `gdb` 中用 `attach` 命令来挂接进程的 `PID` 。并用 `detach` 来取消挂接的进程。

    ```shell
    $ gdb ./test_exec   # 关联要调试的运行的程序

    (gdb) attach 6022   # 挂接进程的 PID
    (gdb) detach        # 取消挂接的进程
    ```

### 2.1.2 调试未运行的程序

先用 `gdb <your_exec_file>` 关联要调试的运行的程序：

```shell
$ gdb ./test_exec     # 关联要调试的运行的程序
```

或者在启动 `gdb` 之后，使用 `exec-file` 命令指定要运行的程序（带路径）：

```shell
$ gdb

(gdb) exec-file ./your_exec_file
```

推荐使用第一种方式，因为使用 `exec-file` 命令虽然能调试程序，但是可能关联不了源代码。

**指定要调试程序的运行参数**

可使用 `run` 命令启动要调试的程序，并指定运行的参数：

```shell
(gdb) run -f -d your_exec_params
```

### 2.1.3 暂停/恢复程序运行

调试程序中，暂停程序运行是必须的， `gdb` 可以方便地暂停程序的运行。你可以设置程序的在哪行停住，在什么条件下停住，在收到什么信号时停往等等。以便于你查看运行时的变量，以及运行时的流程。

当进程被 `gdb` 停住时，你可以使用 `info program` 来查看程序的是否在运行，进程号，被暂停的原因。

在 `gdb` 中，我们可以有以下几种暂停方式：断点（`BreakPoint`）、观察点（`WatchPoint`）、捕捉点（`CatchPoint`）、信号（`Signals`）、线程停止（`Thread Stops`）。如果要恢复程序运行，可以使用 `c` 或是 `continue` 命令。

### 2.1.4 设置断点（BreakPoint）

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

  * set step-mode off

    关闭 step-mode 模式。

当程序被停住了，你可以用 continue 命令恢复程序的运行直到程序结束，或下一个断点到来。也可以使用 step 或 next 命令单步跟踪程序。

* continue [ignore-count]
* c [ignore-count]
* fg [ignore-count]

    恢复程序运行，直到程序结束，或是下一个断点到来。 ignore-count 表示忽略其后的断点次数。 continue，c，fg 三个命令都是一样的意思。

### 2.1.5 设置观察点（WatchPoint）

观察点一般来观察某个表达式（变量也是一种表达式）的值是否有变化了，如果有变化，马上停住程序。我们有下面的几种方法来设置观察点：

* watch \<expr\>

    为表达式（变量）expr 设置一个观察点。一量表达式值有变化时，马上停住程序。

* rwatch \<expr\>

    当表达式（变量）expr 被读时，停住程序。

* awatch \<expr\>

    当表达式（变量）的值被读或被写时，停住程序。

* info watchpoints

    列出当前所设置了的所有观察点。

### 2.1.6 设置捕捉点（CatchPoint）

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

### 2.1.7 维护停止点

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

## 3. print 命令

可使用 `print <var_name>` 查看变量的值，`print` 命令的格式为：

```shell
print [options --] [/fmt] expr
```

或者

```shell
p [options --] [/fmt] expr
```

### 3.1 print 操作符

1. `@`

    是一个和数组有关的操作符，在后面会有更详细的说明。
2. `::`

    指定一个在文件或是一个函数中的变量。
    
3. `{}`

    表示一个指向内存地址的类型为type的一个对象。

### 3.2 察看内容

* 全局变量（所有文件可见的）
* 静态全局变量（当前文件可见的）
* 局部变量（当前 `Scope` 可见的）

如果你的局部变量和全局变量发生冲突（重名），一般情况下，局部变量会覆盖全局变量。如果此时你想查看全局变量的值，可以使用 "`::`" 操作符：

```cpp
file::variable
function::variable
```

例如：

查看文件 `f2.c` 中的全局变量 `x` 的值：

```cpp
(gdb) p 'f2.c'::x
```

注：如果你的程序编译时开启了优化选项，那么在用 `GDB` 调试被优化过的程序时，可能会发生某些变量不能访问，或是取值错误的情况。对付这种情况时，需要在编译程序时关闭编译优化。例如 `GCC`，你可以使用 "`-gstabs`" 选项来解决这个问题。

### 3.3 察看数组

1. 动态数组

    格式：

    ```text
    print array@len

    array: 数组的首地址，len: 数据的长度
    ```

    例如：

    ```shell
    (gdb) print array@len

    $1 = {2, 4, 6, 8, 10}
    ```

2. 静态数组

    可以直接用 "`print 数组名`"，就可以显示数组中所有数据的内容了。

### 3.4 输出格式

打印变量的时候可以设置变量的输出格式，命令格式：

```shell
print [/fmt] expr
```

所有 `/fmt` 格式：

```text
/x 按 十六进制 格式显示
/d 按 十进制 格式显示 有符号整型
/u 按 十六进制 格式显示 无符号整型
/o 按 八进制 格式显示
/t 按 二进制 格式显示
/a 按 十六进制 格式显示
/c 按 字符 格式显示
/f 按 浮点数 格式显示
```

例如：

按十六进制的格式显示变量 `num` 的值：

```shell
(gdb) print/x num

$1 = 0x80001001
```

### 3.5 察看内存

使用 `examine`（简写 `x`）来查看内存地址中的值。

语法：

```text
x/
```

n、f、u 是可选的参数。

* `n` 是一个正整数，表示显示内存的长度，也就是说从当前地址向后显示几个地址的内容。
* `f` 表示显示的格式，参见上面。如果地址所指的是字符串，那么格式可以是 `s`，如果地十是指令地址，那么格式可以是 `i`。
* `u` 表示从当前地址往后请求的字节数，如果不指定的话，`GDB` 默认是 4 个 bytes。`u` 参数可以用下面的字符来代替，`b` 表示单字节，`h` 表示双字节，`w` 表示四字 节，`g` 表示八字节。当我们指定了字节长度后，`GDB` 会从指内存定的内存地址开始，读写指定字节，并把其当作一个值取出来。


例如：

```text
x/3uh 0x54320 ：从内存地址 0x54320 读取内容，h 表示以双字节为一个单位，3 表示三个单位，u 表示按十六进制显示。
```

### 3.6 察看寄存器

1. 要查看寄存器的值，很简单，可以使用如下命令：

    ```shell
    info registers
    ```

2. 查看寄存器的情况。（除了浮点寄存器）

    ```shell
    info all-registers
    ```

3. 查看所有寄存器的情况。（包括浮点寄存器）

    ```shell
    info registers
    ```

4. 查看所指定的寄存器的情况。

    寄存器中放置了程序运行时的数据，比如程序当前运行的指令地址（ip），程序的当前堆栈地址（sp）等等。你同样可以使用 `print` 命令来访问寄存器的情况，只需要在寄存器名字前加一个

### 3.7 display 自动显示的变量

1. 格式：

    ```shell
    display[/i|s] [expression | addr]
    ```

    例如：

    ```shell
    display/i 

    disable display
    enable display
    ```

    `disable` 和 `enalbe` 不删除自动显示的设置，而只是让其失效和恢复。

2. `info display`

    查看 `display` 设置的自动显示的信息。`GDB` 会打出一张表格，向你报告当然调试中设置了多少个自动显示设置，其中包括，设置的编号，表达式，是否 `enable`。

### 3.8 显示格式设置

1. set print address

    ```shell
    set print address on
    ```
    
    打开地址输出，当程序显示函数信息时，`GDB` 会显出函数的参数地址。

2. set print array

    ```shell
    set print array on
    ```

    打开数组显示，打开后当数组显示时，每个元素占一行，如果不打开的话，每个元素则以逗号分隔，默认值为 `off`。

3. set print array-indexes

    ```shell
    set print array-indexes on
    ```

    对于非字符类型数组，在打印数组中每个元素值的同时，是否同时显示每个元素对应的数组下标，默认值为 `off`。

4. set print elements

    ```shell
    set print elements number-of-elements   # 设置这个最大限制数
    set print elements 0                    # 没有限制
    set print elements unlimited            # 没有限制, 跟设置为 0 一样
    ```

    这个选项是设置数组元素的最大显示个数的，如果你的数组太大了，那么就可以指定一个最大长度，当超过这个长度时，`GDB` 就不再往下显示了，默认最多只显示 200 个元素。如果设置为 0，则表示不限制。

5. set print null-stop

    ```shell
    set print null-stop on
    ```

    如果打开了该选项，那么当显示字符串时，遇到结束符则停止显示，默认值为 `off`。

6. set print pretty on

    ```shell
    set print pretty on
    ```

    如果打开了该选项，以更美观的格式（便于阅读）打印结构体变量的值，默认值为 `off`。

    例如：

    ```shell
    $1 = {
    next = 0x0,
    flags = {
        sweet = 1,
        sour = 1
        },
    meat = 0x54 "Pork"
    }
    ```

7. set print union

    ```shell
    set print union on
    ```

    设置显示结构体时，是否显式其中的 `union` (联合体) 数据。

8. set print object

    ```shell
    set print object on
    ```

    在 `C++` 中，如果一个对象指针指向其派生类，如果打开这个选项，`GDB` 会自动按照虚方法调用的规则显示输出，如果关闭这个选项的话，`GDB` 就不管虚函数表了。

## 4. GDB 布局

使用 `gdb` 时，最好加上 "`--tui`" 选项，否则很大可能会出现花屏现象。

```shell
$ gdb --tui <your_exec_file>
```

如果你已经启动了 `gdb`，可使用下面的命令启用 `tui`：

```shell
(gdb) tui enable        # 启用 tui
(gdb) tui disable       # 关闭 tui
(gdb) tui reg           # 显示相关的寄存器状态

(gdb) tui reg general   # 显示标准寄存器的状态
(gdb) tui reg mmx       # 显示 MMX 寄存器的状态
(gdb) tui reg sse       # 显示 SSE 寄存器的状态

# tui reg 可选的参数有：
# sse, mmx, general, float, all, save, restore, vector, system
```

打开 `gdb` 后，可使用如下命令设置 `layout`（布局）：

* 能够实时看到寄存器值的变化

    ```shell
    layout regs         # 显示当前调试状态的寄存器状态
    ```

* 能够看到源代码和对应汇编的关系   

    ```shell
    layout src          # 显示当前调试状态关联的源代码（如果有的话）
    layout asm          # 显示当前调试状态关联的汇编代码
    layout split        # 分屏同时显示源代码，汇编
    ```

    在 `gdb` 中运行 `set disassemble-next-line on`，表示自动反汇编后面要执行的代码。

## 5. 参考文章

* `Linux 下 gdb 单步调试`

    [https://blog.csdn.net/zzymusic/article/details/4815142](https://blog.csdn.net/zzymusic/article/details/4815142)

* `GDB 单步调试汇编`

    [https://www.cnblogs.com/zhangyachen/p/9227037.html](https://www.cnblogs.com/zhangyachen/p/9227037.html)

* `gdb print 详解`

    [https://blog.csdn.net/litanglian9839/article/details/84966813](https://blog.csdn.net/litanglian9839/article/details/84966813)
