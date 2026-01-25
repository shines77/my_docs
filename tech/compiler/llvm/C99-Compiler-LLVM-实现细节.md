# C99 Compiler + LLVM 实现细节

## 简介

我将为您详细列出完整支持 C99 标准的 C 编译器需要实现的各个部分，特别考虑使用 LLVM IR 作为中间表示，并支持 Windows PE 和 Linux ELF 格式。

## 一、编译器前端部分（必须独立实现）

### 1. **词法分析器 (Lexer)**

- Unicode 字符集支持（C99 要求至少支持基本源字符集）
- 处理三字符序列（trigraphs）和双字符序列（digraphs）
- 识别所有 C99 关键字（37个）：

  - `auto`, `break`, `case`, `char`, `const`, `continue`, `default`, `do`
  - `double`, `else`, `enum`, `extern`, `float`, `for`, `goto`, `if`
  - `inline`, `int`, `long`, `register`, `restrict`, `return`, `short`, `signed`
  - `sizeof`, `static`, `struct`, `switch`, `typedef`, `union`, `unsigned`
  - `void`, `volatile`, `while`
  - C99 新增：`_Bool`, `_Complex`, `_Imaginary`, `restrict`, `inline`

- 支持 C99 数值常量格式：

  - 十六进制浮点常量（0x1.0p-10）
  - 长整型后缀（LL, ULL）

- 处理注释（/**/ 和 //）

### 2. **预处理器**

- 完全实现 C99 预处理指令：

  - `#define`（含参数化宏、`__VA_ARGS__`）
  - `#undef`
  - `#include`（支持 `#include_next`）
  - 条件编译：`#if`, `#ifdef`, `#ifndef`, `#else`, `#elif`, `#endif`
  - `#line`
  - `#error`
  - `#pragma` 和 `_Pragma` 操作符

- 预定义宏：

  - `__STDC__ = 1`
  - `__STDC_VERSION__ = 199901L`
  - `__STDC_HOSTED__`
  - `__FILE__`, `__LINE__`, `__DATE__`, `__TIME__`
  - `__func__`（预定义标识符）

- 宏展开规则（避免重复展开等边界情况）

### 3. **语法分析器 (Parser)**

- 实现完整的 C99 语法（ISO/IEC 9899:1999）
- 支持 C99 新增语法：

  - 变量声明可以出现在代码块中的任意位置
  - `//` 行注释
  - 复合字面量（compound literals）
  - 指定初始化器（designated initializers）
  - 灵活数组成员（flexible array members）
  - 变长数组（variable-length arrays, VLA）
  - `restrict` 类型限定符
  - `inline` 函数声明
  - `long long` 和 `unsigned long long`
  - `_Bool`、`_Complex`、`_Imaginary` 类型
  - 复数浮点常量（如 `1.0 + 2.0*I`）
  - 十六进制浮点常量
  - `for` 循环初始声明（C99允许在for循环中声明变量）

### 4. **语义分析器**

- 类型系统：

  - 基本类型：`void`, `_Bool`, `char`, `short`, `int`, `long`, `long long`
  - 浮点：`float`, `double`, `long double`
  - 复数：`float _Complex`, `double _Complex`, `long double _Complex`
  - 虚数：`float _Imaginary`, `double _Imaginary`, `long double _Imaginary`
  - 枚举类型
  - 结构体/联合体类型（含匿名结构体）
  - 数组类型（包括 VLA）
  - 函数类型
  - 指针类型

- 类型兼容性规则
- 隐式和显式类型转换规则
- 作用域规则（块作用域、文件作用域等）
- 链接属性（内部链接、外部链接、无链接）
- 存储期（静态、自动、分配）
- 类型限定符检查（`const`, `volatile`, `restrict`）
- 常量表达式求值（包括编译时求值）

### 5. **中间表示生成（LLVM IR）**

- AST 到 LLVM IR 的转换：

  - 基本类型到 LLVM 类型的映射

  - 复杂类型表示：
    - 结构体/联合体 → LLVM 结构体类型
    - 数组（包括 VLA） → LLVM 数组类型或指针
    - 函数指针 → LLVM 函数指针类型

  - 控制流转换：
    - `if`/`else` → `br` 指令
    - `switch` → `switch` 指令
    - 循环 → `br` 和 `phi` 指令

  - 表达式转换：
    - 算术运算 → LLVM 算术指令
    - 逻辑运算 → LLVM 整数运算和比较指令
    - 位运算 → LLVM 位运算指令

  - 内存操作：
    - 变量声明 → `alloca` 指令
    - 赋值 → `store` 指令
    - 取值 → `load` 指令
    - 地址获取 → `getelementptr` 指令

  - 函数调用：
    - 处理可变参数函数（`va_list`, `va_start`, `va_arg`, `va_end`, `va_copy`）
    - `inline` 函数处理

- 特殊 C99 特性支持：

  - 变长数组（VLA）处理：
    - 使用 `alloca` 或动态分配
    - 处理 VLA 参数

  - 复合字面量转换
  - 指定初始化器展开
  - 灵活数组成员处理
  - `restrict` 指针优化提示
  - 复数运算转换

- 生成优化友好的 LLVM IR：

  - 正确的 SSA 形式
  - 适当的元数据（调试信息、TBAA 等）

## 二、标准库实现

### 1. **必需的头文件（24个）**

- `<assert.h>` - 断言宏
- `<complex.h>` - 复数运算（可选但完整实现需要）
- `<ctype.h>` - 字符分类和转换
- `<errno.h>` - 错误代码
- `<fenv.h>` - 浮点环境控制
- `<float.h>` - 浮点数特性
- `<inttypes.h>` - 精确宽度整数格式化
- `<iso646.h>` - 替代运算符宏（C95引入，C99包含）
- `<limits.h>` - 整数类型范围
- `<locale.h>` - 本地化类别
- `<math.h>` - 数学函数（包括C99新增如`fma`, `round`, `trunc`等）
- `<setjmp.h>` - 非本地跳转
- `<signal.h>` - 信号处理
- `<stdarg.h>` - 可变参数处理
- `<stdbool.h>` - 布尔类型
- `<stddef.h>` - 通用定义
- `<stdint.h>` - 精确宽度整数类型
- `<stdio.h>` - 标准I/O（含C99新增如`snprintf`, `vfscanf`等）
- `<stdlib.h>` - 通用工具（内存分配、随机数等）
- `<string.h>` - 字符串函数
- `<tgmath.h>` - 泛型数学宏
- `<time.h>` - 时间函数
- `<wchar.h>` - 宽字符处理（C95引入，C99包含）
- `<wctype.h>` - 宽字符分类（C95引入，C99包含）

### 2. **库函数实现要点**

- 数学库函数（必须正确处理边界情况和特殊值）
- I/O 函数（文件操作、格式化I/O等）
- 内存管理（`malloc`, `free`, `calloc`, `realloc`）
- 字符串和内存操作
- 多字节/宽字符转换
- 区域设置相关函数
- 复数运算函数
- 浮点环境控制
- 精确宽度整数运算

## 三、LLVM 后端集成

### 1. **目标平台支持配置**

```cpp
// Windows (PE) 配置
TargetTriple = "x86_64-pc-windows-msvc";  // 或 "i686-pc-windows-msvc"
DataLayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128";

// Linux (ELF) 配置
TargetTriple = "x86_64-pc-linux-gnu";  // 或 "i686-pc-linux-gnu"
DataLayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128";
```

### 2. **ABI 兼容性处理**

- **Windows x64 ABI**：

  - 调用约定：Microsoft x64 调用约定（前4个参数在 RCX, RDX, R8, R9）
  - 异常处理：SEH（结构化异常处理）
  - 堆栈对齐：16字节对齐
  - 名称修饰：`_` 前缀（C语言）
  
- **Linux x64 ABI**：

  - 调用约定：System V AMD64 ABI（前6个整数参数在 RDI, RSI, RDX, RCX, R8, R9）
  - 浮点参数：XMM0-XMM7
  - 堆栈对齐：16字节对齐
  - 异常处理：DWARF 异常处理

### 3. **运行时环境集成**

- **Windows**：

  - 链接到 UCRT（Universal C Runtime）或 MSVCRT
  - 入口点：`mainCRTStartup`（调用 `main`）
  - 需要实现 `_chkstk` 堆栈检查
  - TLS（线程局部存储）支持
  
- **Linux**：

  - 链接到 glibc 或 musl
  - 入口点：`_start`（调用 `__libc_start_main`，再调用 `main`）
  - 需要支持 `.init_array` 和 `.fini_array`

## 四、平台特定需求

### **Windows PE 格式支持**

1. **对象文件生成**：

   - COFF（Common Object File Format）格式
   - `.text`, `.data`, `.bss`, `.rdata` 节
   - 调试信息：CodeView 格式

2. **PE 可执行文件生成**：

   - DOS 存根
   - PE 头部：`IMAGE_NT_HEADERS`
   - 节表：至少需要 `.text`, `.data`, `.rdata`
   - 导入表（IAT）和导出表
   - 资源节（可选）
   - 重定位表（用于 DLL）
   - TLS 目录

3. **运行时支持**：

   - CRT（C Runtime）初始化
   - 结构化异常处理（SEH）
   - DLL 支持（`__declspec(dllimport/dllexport)`）
   - `__stdcall`, `__cdecl`, `__fastcall` 调用约定

### **Linux ELF 格式支持**

1. **对象文件生成**：

   - ELF 格式（Executable and Linkable Format）
   - `.text`, `.data`, `.bss`, `.rodata` 节
   - 符号表、字符串表、重定位表
   - 调试信息：DWARF 格式

2. **ELF 可执行文件生成**：

   - ELF 头部：`e_ident`, `e_type`, `e_machine`, `e_entry`
   - 程序头表（Program Headers）：`PT_LOAD`, `PT_DYNAMIC` 等
   - 节头表（Section Headers）
   - 动态链接信息：`.dynamic` 节
   - GOT（全局偏移表）和 PLT（过程链接表）
   - 重定位节

3. **运行时支持**：

   - 动态链接器支持（`ld-linux.so`）
   - `LD_PRELOAD` 机制
   - TLS（线程局部存储）实现
   - 信号处理框架
   - `atexit` 处理

## 五、链接器功能

### 1. **静态链接**

- 符号解析和重定位
- 节合并
- 库搜索路径处理（`-L` 和 `-l` 选项）

### 2. **动态链接（共享库）**

- **Windows DLL**：

  - 导出函数表
  - 导入地址表（IAT）
  - 延迟加载
  - 清单文件支持
  
- **Linux 共享对象**：

  - 符号版本控制
  - 符号可见性控制（`-fvisibility`）
  - 动态段（`.dynamic`）
  - `DT_NEEDED` 条目

### 3. **链接器优化**

- 函数级别链接（Windows）
- 垃圾收集未使用节
- 相同内容合并（`-fmerge-constants`）
- 增量链接支持

## 六、调试信息生成

### 1. **Windows (CodeView)**

- 生成 PDB（Program Database）文件
- 行号信息
- 局部变量和类型信息
- 支持 WinDbg 和 Visual Studio 调试器

### 2. **Linux (DWARF)**

- DWARF 调试信息格式（通常 DWARF-4）
- `.debug_info`, `.debug_line`, `.debug_abbrev` 等节
- 支持 GDB 和 LLDB 调试器

## 七、构建系统集成

### 1. **编译器驱动程序**

- 命令行参数解析
- 多阶段编译控制（预处理→编译→汇编→链接）
- 依赖文件生成（`-M`, `-MM` 选项）
- 响应文件支持

### 2. **工具链组件**

- 汇编器（生成目标文件）
- 链接器（或调用系统链接器）
- 归档工具（`ar`，用于静态库）
- 库工具（`lib` 或 `dlltool`）

## 八、测试与验证

### 1. **符合性测试套件**

- 使用 Perennial 或 Plum Hall C99 测试套件
- GCC  torture 测试
- 针对边界情况的专门测试

### 2. **平台兼容性测试**

- Windows 版本兼容性（XP 到 Windows 11）
- Linux 发行版兼容性（glibc 版本）
- 架构兼容性（x86, x86_64）

## 九、优化支持

### 1. **编译器优化**

- 常量传播和折叠
- 死代码消除
- 循环优化
- 内联扩展
- 尾调用优化

### 2. **LLVM 优化通道**

```cpp
// 典型的优化管道
PassManagerBuilder Builder;
Builder.OptLevel = 2;
Builder.SizeLevel = 0;
Builder.Inliner = createFunctionInliningPass(275);
Builder.populateModulePassManager(MPM);
```

## 十、文档和工具

### 1. **用户文档**

- 编译器用户手册
- 语言扩展文档
- 平台特定行为文档

### 2. **开发者工具**

- 编译器自举能力
- 交叉编译支持
- 性能分析工具集成

---

## **完整实现路线图**

### 阶段1：基础前端

1. 词法分析器和预处理器
2. 语法分析器（生成 AST）
3. 基本类型系统
4. 简单表达式和语句的 LLVM IR 生成

### 阶段2：完整语言支持

1. 完整 C99 语法支持
2. 复杂类型（结构体、联合体、数组）
3. 函数和指针
4. 标准库头文件

### 阶段3：平台后端

1. x86_64 LLVM 后端集成
2. Windows PE 支持
3. Linux ELF 支持
4. 调试信息生成

### 阶段4：优化和工具

1. 编译器优化
2. 链接器支持
3. 工具链完成
4. 文档和测试

---

**注意**：这是一个极其庞大的工程。实际上，大多数编译器项目（如 Clang）需要数百人年的工作量。建议从 C99 子集开始，逐步扩展功能，或者基于现有开源编译器（如 TinyCC）进行扩展，而不是从零开始。

## 问题

一个完整支持 C 语言 C99 标准的 C 编译器都需要完成哪些部分，IR 表示层可以用 LLVM 来实现，并通过链接器编译为可执行程序，需要同时兼容 Windows 的 PE 格式和 Linux 的 ELF 格式，帮我分别列出来。
