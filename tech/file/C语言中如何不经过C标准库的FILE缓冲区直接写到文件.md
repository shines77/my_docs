# C 语言中如何不经过 C 标准库的 FILE 缓冲区直接写到文件

## 1. 问题

C 语言中有不经过 C 标准库的 FILE 缓冲区直接写到文件的函数吗？

## 2. 回答

在C语言中，可以通过以下方式绕过标准库的缓冲区（`FILE`结构体）直接写入文件：

---

### **1. 使用操作系统提供的系统调用（低级I/O函数）**

这些函数直接通过**文件描述符**（而非`FILE`指针）操作文件，不经过C标准库的缓冲区：

#### **Unix/Linux系统**

```c
#include <fcntl.h>
#include <unistd.h>

int main() {
    // 打开文件（系统调用）
    int fd = open("file.txt", O_WRONLY | O_CREAT, 0644);
    if (fd == -1) {
        perror("open failed");
        return 1;
    }

    // 直接写入数据（无缓冲）
    char *data = "Hello, World!";
    ssize_t bytes_written = write(fd, data, strlen(data));
    if (bytes_written == -1) {
        perror("write failed");
    }

    // 关闭文件
    close(fd);
    return 0;
}
```
- **关键函数**：

  - `open()`：打开文件，返回文件描述符。
  - `write()`：直接写入数据到文件描述符。
  - `close()`：关闭文件描述符。

#### **Windows系统**

```c
#include <windows.h>

int main() {
    HANDLE hFile = CreateFile(
        "file.txt",
        GENERIC_WRITE,
        0,
        NULL,
        CREATE_ALWAYS,
        FILE_ATTRIBUTE_NORMAL,
        NULL
    );
    if (hFile == INVALID_HANDLE_VALUE) {
        return 1;
    }

    char *data = "Hello, World!";
    DWORD bytesWritten;
    WriteFile(hFile, data, strlen(data), &bytesWritten, NULL);

    CloseHandle(hFile);
    return 0;
}
```
- **关键函数**：

  - `CreateFile()`：打开或创建文件。
  - `WriteFile()`：直接写入数据到文件句柄。
  - `CloseHandle()`：关闭文件句柄。

---

### **2. 使用C标准库的无缓冲模式（仍使用`FILE`，但禁用缓冲）**

虽然仍涉及`FILE`结构体，但通过设置无缓冲模式，每次操作直接传递到操作系统：

```c
#include <stdio.h>
#include <string.h>

int main() {
    FILE *fp = fopen("file.txt", "w");
    if (!fp) {
        perror("fopen failed");
        return 1;
    }

    // 设置为无缓冲模式
    setvbuf(fp, NULL, _IONBF, 0);

    char *data = "Hello, World!";
    fwrite(data, 1, strlen(data), fp);

    fclose(fp);
    return 0;
}
```

- **关键函数**：

  - `setvbuf(fp, NULL, _IONBF, 0)`：禁用缓冲区，每次`fwrite`直接调用系统级写入。

---

### **3. 直接系统调用与标准库缓冲的区别**

| **特性**               | **系统调用（`write`）**         | **标准库无缓冲模式（`setvbuf`）** |
|------------------------|--------------------------------|----------------------------------|
| **依赖**               | 操作系统API                   | C标准库                          |
| **跨平台性**           | 差（需平台适配）              | 好（标准C语法）                  |
| **缓冲层级**           | 绕过用户空间缓冲，可能仍有内核缓冲 | 用户空间无缓冲，内核可能仍有缓冲 |
| **性能**               | 高频小数据写入性能较差        | 同左，但代码更简洁               |
| **典型场景**           | 需要完全控制I/O的底层操作      | 希望保持标准接口但禁用缓冲       |

---

### **4. 如何确保数据真正写入磁盘？**

即使绕过了用户空间缓冲区，操作系统内核可能仍有缓存。若需强制同步写入硬件，需额外操作：

#### **Unix/Linux**

```c
// 打开文件时添加同步标志
int fd = open("file.txt", O_WRONLY | O_CREAT | O_SYNC, 0644);

// 或调用 fsync()
write(fd, data, len);
fsync(fd); // 强制将内核缓冲区刷入磁盘
```

#### **Windows**

```c
// 创建文件时指定标志
HANDLE hFile = CreateFile(
    "file.txt",
    GENERIC_WRITE,
    0,
    NULL,
    CREATE_ALWAYS,
    FILE_FLAG_WRITE_THROUGH,  // 同步写入
    NULL
);

// 或调用 FlushFileBuffers()
WriteFile(hFile, data, len, &bytesWritten, NULL);
FlushFileBuffers(hFile);
```

---

### **总结**

- **完全绕过`FILE`缓冲区**：使用操作系统提供的系统调用（如Unix的`write`或Windows的`WriteFile`）。
- **禁用标准库缓冲**：使用`setvbuf(fp, NULL, _IONBF, 0)`，但依然依赖`FILE`结构体。
- **数据持久化**：若需确保数据写入物理设备，需额外调用同步函数（如`fsync`或`FlushFileBuffers`）。

根据需求选择合适方案，平衡性能、控制力与代码可移植性。
