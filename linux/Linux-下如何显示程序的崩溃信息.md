# Linux 下如何显示程序的崩溃信息

## 问题

> Linux 下如何让程序显示崩溃时的信息？

## 回答

在 Linux 下，如果程序崩溃，可以通过以下方法显示崩溃时的详细信息，帮助定位问题：

---

### 1. **使用 `gdb` 调试工具**

`gdb` 是 Linux 下强大的调试工具，可以捕获程序崩溃时的堆栈信息。

#### 步骤：

1. 编译程序时加上调试信息（使用 `-g` 选项）：

   ```bash
   gcc -g -o my_program my_program.c
   ```

2. 使用 `gdb` 运行程序：

   ```bash
   gdb ./my_program
   ```

3. 在 `gdb` 中运行程序：

   ```bash
   (gdb) run
   ```

4. 如果程序崩溃，`gdb` 会自动停在崩溃点。输入以下命令查看堆栈信息：

   ```bash
   (gdb) bt
   ```

   `bt`（backtrace）会显示函数调用堆栈，帮助定位崩溃位置。

5. 如果需要更详细的信息，可以查看寄存器和内存：

   ```bash
   (gdb) info registers
   (gdb) x/10x $sp
   ```

---

### 2. **使用 `core dump` 文件**

Core dump 是程序崩溃时的内存镜像，包含崩溃时的完整状态。

#### 步骤：

1. 启用 core dump：

   - 检查当前 core dump 设置：

     ```bash
     ulimit -c
     ```

     如果显示 `0`，表示未启用。

   - 启用 core dump：

     ```bash
     ulimit -c unlimited
     ```

   - 永久生效：将 `ulimit -c unlimited` 添加到 `~/.bashrc` 或 `/etc/profile`。

2. 设置 core dump 文件路径：

   - 编辑 `/etc/sysctl.conf`，添加：

     ```bash
     kernel.core_pattern = /tmp/core-%e-%p-%t
     ```

   - 使配置生效：

     ```bash
     sysctl -p
     ```

3. 运行程序，等待崩溃生成 core dump 文件。

4. 使用 `gdb` 分析 core dump：

   ```bash
   gdb ./my_program /tmp/core-<filename>
   ```

   然后使用 `bt` 查看堆栈信息。

---

### 3. **使用 `strace` 跟踪系统调用**

`strace` 可以跟踪程序的系统调用和信号，帮助分析崩溃原因。

#### 步骤：

1. 使用 `strace` 运行程序：

   ```bash
   strace ./my_program
   ```

2. 如果程序崩溃，`strace` 会显示最后的系统调用和信号。

3. 结合 `strace` 输出和代码逻辑，定位问题。

---

### 4. **使用 `valgrind` 检测内存问题**

`valgrind` 可以检测内存泄漏、非法内存访问等问题。

#### 步骤：

1. 使用 `valgrind` 运行程序：

   ```bash
   valgrind --leak-check=full ./my_program
   ```

2. 如果程序崩溃，`valgrind` 会显示详细的错误信息，包括内存问题的位置。

---

### 5. **捕获信号并打印堆栈**

在程序中捕获信号（如 `SIGSEGV`），并打印堆栈信息。

#### 示例代码：

```c
#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <execinfo.h>

void print_stacktrace() {
    void *buffer[100];
    int size = backtrace(buffer, 100);
    char **strings = backtrace_symbols(buffer, size);
    if (strings == NULL) {
        perror("backtrace_symbols");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < size; i++) {
        printf("%s\n", strings[i]);
    }
    free(strings);
}

void signal_handler(int sig) {
    fprintf(stderr, "Error: signal %d\n", sig);
    print_stacktrace();
    exit(1);
}

int main() {
    signal(SIGSEGV, signal_handler);  // 捕获段错误
    signal(SIGABRT, signal_handler);  // 捕获异常终止

    // 触发崩溃的代码
    int *ptr = NULL;
    *ptr = 10;

    return 0;
}
```

#### 编译和运行：

```bash
gcc -g -o my_program my_program.c -rdynamic
./my_program
```

---

### 6. **查看系统日志**

如果程序崩溃未生成 core dump 或调试信息，可以检查系统日志：

```bash
dmesg | grep -i segfault
```

---

### 总结

- 使用 `gdb` 或 `core dump` 是最常用的方法。
- 结合 `strace` 和 `valgrind` 可以进一步分析问题。
- 在代码中捕获信号并打印堆栈信息，可以快速定位崩溃位置。
