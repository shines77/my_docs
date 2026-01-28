# LLVM v10.0.1 构建指南 (Windows)

本指南将帮助您在 Windows 上使用 Visual Studio 2017 构建 LLVM v10.0.1。

## 前提条件

- ✅ Visual Studio 2017 (已安装)
- ✅ CMake 3.18.1 (已安装)
- ✅ LLVM 源代码位于 `C:\llvm`
- 需要：至少 50GB 可用磁盘空间
- 需要：至少 16GB RAM (推荐 32GB)

## 构建步骤

### 1. 创建构建目录

```powershell
# 在 C:\llvm 目录下创建 build 目录
cd C:\llvm
mkdir build
cd build
```

### 2. 配置 CMake (Release 构建，推荐)

对于日常开发使用，推荐构建 **Release** 版本（速度快，体积小）：

```powershell
cmake -G "Visual Studio 15 2017" -A x64 `
  -DCMAKE_BUILD_TYPE=Release `
  -DLLVM_ENABLE_PROJECTS="clang" `
  -DLLVM_TARGETS_TO_BUILD="X86" `
  -DLLVM_INCLUDE_TESTS=OFF `
  -DLLVM_INCLUDE_EXAMPLES=OFF `
  -DLLVM_ENABLE_ASSERTIONS=OFF `
  ..
```

**参数说明：**
- `-G "Visual Studio 15 2017" -A x64`: 使用 VS2017 生成 64 位项目
- `-DCMAKE_BUILD_TYPE=Release`: Release 模式（优化编译）
- `-DLLVM_ENABLE_PROJECTS="clang"`: 只构建 LLVM 和 Clang（可选，减少构建时间）
- `-DLLVM_TARGETS_TO_BUILD="X86"`: 只构建 X86 目标（减少构建时间）
- `-DLLVM_INCLUDE_TESTS=OFF`: 不构建测试（节省时间）
- `-DLLVM_INCLUDE_EXAMPLES=OFF`: 不构建示例
- `-DLLVM_ENABLE_ASSERTIONS=OFF`: Release 模式下关闭断言

### 3. 开始构建

```powershell
# 使用多核编译（根据您的 CPU 核心数调整 -m 参数）
cmake --build . --config Release -j 8
```

**注意：**

- 首次构建可能需要 **2-4 小时**（取决于硬件配置）
- 如果内存不足，可以减少并行任务数：`-j 4` 或 `-j 2`
- 构建过程会占用大量磁盘空间（约 30-40GB）

### 4. 验证构建结果

构建完成后，检查关键文件是否存在：

```powershell
# 检查 LLVMConfig.cmake
Test-Path C:\llvm\build\lib\cmake\llvm\LLVMConfig.cmake

# 检查 LLVM 库
dir C:\llvm\build\Release\lib\LLVM*.lib
```

## 可选：Debug 构建

如果您需要调试 LLVM 本身，可以构建 Debug 版本（**不推荐用于日常开发**）：

```powershell
cmake -G "Visual Studio 15 2017" -A x64 `
  -DCMAKE_BUILD_TYPE=Debug `
  -DLLVM_ENABLE_PROJECTS="clang" `
  -DLLVM_TARGETS_TO_BUILD="X86" `
  -DLLVM_INCLUDE_TESTS=OFF `
  -DLLVM_INCLUDE_EXAMPLES=OFF `
  ..

cmake --build . --config Debug -j 4
```

**警告：** Debug 构建会非常慢，且占用更多空间（60GB+）。

## 构建后配置

构建完成后，更新 jlang-c 项目的 CMakeLists.txt：

```cmake
# 设置 LLVM_DIR 指向构建目录
set(LLVM_DIR "C:/llvm/build/lib/cmake/llvm" CACHE PATH "LLVM CMake directory")
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
cmake --build .
```
