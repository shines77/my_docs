# LLVM v10.0.1 构建指南 (Windows)

本指南将帮助您在 Windows 上使用 Visual Studio 2017 构建 LLVM v10.0.1。

## 前提条件

- ✅ Visual Studio 2017 (已安装)
- ✅ CMake 3.18.1 (已安装)
- ✅ LLVM 源代码位于 `C:\llvm-build\llvm-project\llvm`
- 需要：至少 50GB 可用磁盘空间
- 需要：至少 16GB RAM (推荐 32GB)

## 构建步骤

### 1. 创建构建目录

```powershell
# 在 C:\llvm 目录下创建 build 目录
cd C:\llvm-build
mkdir build
cd build
```

### 2. 配置 CMake (Release 构建，推荐)

如果使用 CMake 4.0 以上的版本，需要先手动把 CMakeLists.txt 中的 cmake_minimum_required(VERSION 3.4.3) 改为：

```bash
# 必须大于 3.5 版本
cmake_minimum_required(VERSION 3.5)

# 或者这种新的奇怪的方式，也许能解决报错的问题，如果还是报错，可以尝试把3.5和3.28对调一下
cmake_minimum_required(VERSION 3.5..3.28)
```

因为 4.0 以后，最低支持的版本是 3.5，低于这个版本会报错。

对于日常开发使用，推荐构建 **Release** 版本（速度快，体积小）：

```powershell
## MSVC 2017 (x64)
cmake ../llvm-project/llvm `
  -G "Visual Studio 15 2017" -A x64 -Thost=x64 `
  -DCMAKE_BUILD_TYPE=Release `
  -DCMAKE_INSTALL_PREFIX="C:/llvm-build/install" `
  -DLLVM_ENABLE_PROJECTS="llvm;clang" `
  -DLLVM_TARGETS_TO_BUILD="X86" `
  -DLLVM_INCLUDE_TESTS=OFF `
  -DLLVM_INCLUDE_EXAMPLES=OFF `
  -DLLVM_ENABLE_ASSERTIONS=OFF `
  -DLLVM_OPTIMIZED_TABLEGEN=ON `
  -DBUILD_SHARED_LIBS=OFF `
  -DLLVM_USE_CRT_RELEASE=MT `
  -DCMAKE_POLICY_VERSION_MINIMUM="3.5" `
  -DCMAKE_POLICY_DEFAULT_CMP0000=OLD `
  -DCMAKE_POLICY_DEFAULT_CMP0053=NEW `
  -Wno-dev

## mingw-64
cmake ../llvm-project/llvm `
  -G "MinGW Makefiles" `
  -DCMAKE_SYSTEM_PROCESSOR=x86_64 `
  -DCMAKE_INSTALL_PREFIX="C:/llvm-build/install" `
  -DLLVM_ENABLE_PROJECTS="llvm;clang" `
  -DLLVM_TARGETS_TO_BUILD="X86" `
  -DCMAKE_BUILD_TYPE=Release `
  -DLLVM_OPTIMIZED_TABLEGEN=ON `
  -DBUILD_SHARED_LIBS=OFF `
  -DLLVM_USE_CRT_RELEASE=MT `
  -DCMAKE_POLICY_VERSION_MINIMUM="3.5" `
  -DCMAKE_POLICY_DEFAULT_CMP0000=NEW `
  -DCMAKE_POLICY_DEFAULT_CMP0053=NEW `
  -Wno-dev
```

**参数说明：**

- `-G "Visual Studio 15 2017" -A x64`: 使用 VS2017 生成 64 位项目
- `-Thost=x64`: **极其重要！** 这告诉 VS 使用 64 位的编译器和链接器来编译 LLVM。如果不加这个，链接时极大概率会报 Out of Heap Space 内存溢出错误。
- `-DCMAKE_BUILD_TYPE=Release`: Release 模式（优化编译）
- `-DCMAKE_INSTALL_PREFIX="C:/llvm/install"`：指定安装路径，默认安装路径为 "C:\Program Files(x86)\LLVM"，不推荐用默认路径。
- `-DLLVM_ENABLE_PROJECTS="clang"`: 只构建 LLVM 和 Clang（可选，减少构建时间）
- `-DLLVM_TARGETS_TO_BUILD="X86"`: 只构建 X86 目标（减少构建时间）
- `-DLLVM_INCLUDE_TESTS=OFF`: 不构建测试（节省时间）
- `-DLLVM_INCLUDE_EXAMPLES=OFF`: 不构建示例
- `-DLLVM_ENABLE_ASSERTIONS=OFF`: Release 模式下关闭断言
- `-DLLVM_OPTIMIZED_TABLEGEN=ON`: 优化TableGen
- `-DBUILD_SHARED_LIBS=OFF`: 构建静态库（可选）
- `-DLLVM_USE_CRT_RELEASE=MT`: 使用静态运行时

如果你的 MinGW64 路径不在系统 PATH 中，可以显式指定编译器：

```bash
  -DCMAKE_C_COMPILER="C:/mingw64/bin/gcc.exe" `
  -DCMAKE_CXX_COMPILER="C:/mingw64/bin/g++.exe" `
  -DCMAKE_ASM_COMPILER="C:/mingw64/bin/gcc.exe" `
```

这两个选项不太推荐，有可能会覆盖掉其他编译选项。推荐使用 `-DCMAKE_SYSTEM_PROCESSOR=x86_64` 。

```bash
# 通过 CMAKE_SYSTEM_PROCESSOR，推荐
cmake .. -G "MinGW Makefiles" -DCMAKE_SYSTEM_PROCESSOR=x86_64

# 或使用自定义选项（不推荐，可能会覆盖其他编译选项）
cmake .. -G "MinGW Makefiles" \
  -DCMAKE_CXX_FLAGS="-m64" \
  -DCMAKE_C_FLAGS="-m64"
```

### 3. 开始构建

```powershell
# 使用多核编译（根据您的 CPU 核心数调整 -m 参数）
cmake --build . --config Release -j 8
```

在 x86 上需要特别构建：

```powershell
cmake --build . --config Release --target LLVMX86CodeGen -j 8
```

**注意：**

- 首次构建可能需要 **2-4 小时**（取决于硬件配置）
- 如果内存不足，可以减少并行任务数：`-j 4` 或 `-j 2`
- 构建过程会占用大量磁盘空间（约 30-40GB）

只构建 Clang，不构建 LLVM 除了基础库的其他东西：

```bash
cmake --build . --config Release --target clang -j 8
```

- 注意: 这里我们指定 --target clang，这样只会编译 Clang 及其依赖，而不会编译 LLVM 文档和其他杂项，能节省时间。
- 这个过程根据 CPU 核心数，可能需要 1 到 3 小时。

### 4. 验证构建结果

构建完成后，检查关键文件是否存在：

```powershell
# 检查 LLVMConfig.cmake
Test-Path C:\llvm-build\build\lib\cmake\llvm\LLVMConfig.cmake

# 检查 LLVM 库
dir C:\llvm-build\build\Release\lib\LLVM*.lib
```

### 5. 安装 LLVM 和 Clang

```bash
cmake --build . --config Release --target install -j 8
```

## 可选：Debug 构建

如果您需要调试 LLVM 本身，可以构建 Debug 版本（**不推荐用于日常开发**）：

```powershell
cmake ../llvm-project/llvm `
  -G "Visual Studio 15 2017" -A x64 -Thost=x64 `
  -DCMAKE_BUILD_TYPE=Debug `
  -DLLVM_ENABLE_PROJECTS="llvm,clang" `
  -DLLVM_TARGETS_TO_BUILD="X86" `
  -DLLVM_INCLUDE_TESTS=OFF `
  -DLLVM_INCLUDE_EXAMPLES=OFF `
  -DLLVM_ENABLE_ASSERTIONS=ON `
  -DLLVM_OPTIMIZED_TABLEGEN=ON `
  -DBUILD_SHARED_LIBS=OFF `
  -DLLVM_USE_CRT_RELEASE=MT `
  -DCMAKE_POLICY_VERSION_MINIMUM="3.5" `
  -DCMAKE_POLICY_DEFAULT_CMP0000=NEW `
  -DCMAKE_POLICY_DEFAULT_CMP0053=NEW `
  -Wno-dev

cmake --build . --config Debug -j 8
```

**警告：** Debug 构建会非常慢，且占用更多空间（60GB+）。

## 构建后配置

构建完成后，更新 jlang-c 项目的 CMakeLists.txt：

```cmake
# 设置 LLVM_DIR 指向构建目录
set(LLVM_DIR "C:/llvm-build/build/lib/cmake/llvm" CACHE PATH "LLVM CMake directory")
```

## 故障排除

### 问题 1: 内存不足

**症状：** 编译器崩溃或系统变慢

**解决方案：**

- 减少并行任务数：`-j 2` 或 `-j 1`
- 关闭其他应用程序
- 考虑增加虚拟内存

### 问题 2: 磁盘空间不足

**症状：** 构建失败，提示磁盘空间不足

**解决方案：**

- 清理 C 盘空间
- 或将构建目录移到其他盘（需修改路径）

### 问题 3: 构建时间过长

**解决方案：**

- 只构建必要的组件（已在上述配置中优化）
- 使用 SSD 硬盘
- 升级硬件配置

## 预计资源消耗

| 配置 | 时间 | 磁盘空间 | 内存 |
|------|------|----------|------|
| Release (推荐) | 2-3 小时 | ~35GB | 8-16GB |
| Debug | 4-6 小时 | ~60GB | 16-32GB |

## 下一步

构建完成后，返回 jlang-c 项目：

```powershell
cd C:\Project\VibeCoding\antigravity\jlang-c\build
cmake ..
cmake --build . --config Release
```
