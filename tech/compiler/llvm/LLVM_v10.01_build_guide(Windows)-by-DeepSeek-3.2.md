# LLVM v10.01 build guide (Windows)

Answer by DeepSeek-3.2

我将为您提供在 Windows 上使用 CMake 和 Visual Studio 2017 构建 LLVM 和 Clang 10.0.1 的详细教程。

## 系统要求

- Windows 10 或更高版本
- Visual Studio 2017（建议使用15.9或更高版本）
- 至少30GB可用磁盘空间
- 至少16GB RAM（建议32GB）
- Python 3.6或更高版本

## 第1步：安装必要软件

### 1.1 安装 Visual Studio 2017

- 确保安装以下组件：

  - "Desktop development with C++"
  - Windows 10 SDK
  - C++ CMake tools for Windows

### 1.2 安装 Python

- 下载 Python 3.6+（建议3.8）
- 安装时勾选 "Add Python to PATH"

### 1.3 安装 Git

- 下载Git for Windows：[https://git-scm.com/]()
- 安装时选择 "Use Git from the Windows Command Prompt"

### 1.4 安装 CMake

- 下载CMake 3.13+：[https://cmake.org/download/]()
- 安装时选择 "Add CMake to the system PATH"

## 第2步：获取源代码

由于 10.0.1 属于 Monorepo 时代，我们需要下载包含 LLVM、Clang 等所有项目的统一源码包。

```bash
# 打开PowerShell或命令提示符（以管理员身份运行）
# 创建工作目录

mkdir C:\llvm-build
cd C:\llvm-build

# 克隆LLVM源代码（使用--depth 1加速下载）

git clone --depth 1 --branch llvmorg-10.0.1 https://github.com/llvm/llvm-project.git
或者
git checkout llvmorg-10.0.1

# 进入源代码目录

cd llvm-project
```

## 第3步：创建构建目录结构

```bash
# 返回上级目录
cd ..

# 创建构建目录和安装目录

mkdir build
mkdir install

# 目录结构：

# C:\llvm-build\
#   ├── llvm-project\    # 源代码
#   ├── build\           # 构建目录
#   └── install\         # 安装目录
```

## 第4步：配置CMake

### 4.1 使用 CMake GUI 配置（推荐新手）

1. 打开 CMake GUI
2. 设置源代码路径：`C:/llvm-build/llvm-project/llvm`
3. 设置构建路径：`C:/llvm-build/build`
4. 点击 "Configure"
5. 选择 Visual Studio 2017 作为生成器
   - 选择 "Visual Studio 15 2017"
   - 对于64位构建，选择 "Visual Studio 15 2017 Win64"
6. 设置以下 CMake 变量：

| 变量名 | 值 | 说明 |
|--------|-----|------|
| CMAKE_INSTALL_PREFIX | C:/llvm-build/install | 安装路径 |
| LLVM_ENABLE_PROJECTS | clang;clang-tools-extra | 启用clang项目 |
| LLVM_TARGETS_TO_BUILD | X86 | 简化构建，只构建x86目标 |
| CMAKE_BUILD_TYPE | Release | 发布版本 |
| LLVM_OPTIMIZED_TABLEGEN | ON | 优化TableGen |
| BUILD_SHARED_LIBS | OFF | 构建静态库（可选） |
| LLVM_USE_CRT_RELEASE | MT | 使用静态运行时 |

1. 点击"Configure"直到所有红色消失
2. 点击"Generate"

### 4.2 或者使用命令行配置

```bash
cd C:\llvm-build\build

# 基本配置
cmake ../llvm-project/llvm ^
  -G "Visual Studio 15 2017 Win64" -A x64 -Thost=x64 ^
  -DCMAKE_INSTALL_PREFIX="C:/llvm-build/install" ^
  -DLLVM_ENABLE_PROJECTS="clang;clang-tools-extra" ^
  -DLLVM_TARGETS_TO_BUILD="X86" ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DLLVM_OPTIMIZED_TABLEGEN=ON ^
  -DBUILD_SHARED_LIBS=OFF ^
  -DLLVM_USE_CRT_RELEASE=MT
```

## 第5步：构建 LLVM/Clang

### 5.1 使用 Visual Studio 构建

```bash
# 打开生成的解决方案
devenv C:\llvm-build\build\LLVM.sln
```

在Visual Studio中：

1. 在解决方案配置中选择 "Release"
2. 右键点击"ALL_BUILD" -> "生成"
3. 构建可能需要数小时，取决于您的硬件

### 5.2 或者使用命令行构建

```bash
cd C:\llvm-build\build

# 使用所有CPU核心构建
cmake --build . --config Release --target ALL_BUILD -- /m /p:Platform=x64

# 或者指定并行构建的作业数
cmake --build . --config Release --target ALL_BUILD --parallel 8
```

## 第6步：安装构建结果

```bash
# 安装到指定目录
cmake --build . --config Release --target INSTALL
```

## 第7步：验证安装

```bash
# 测试clang是否正常工作
C:\llvm-build\install\bin\clang --version

# 应该显示类似以下内容：
# clang version 10.0.1
# Target: x86_64-pc-windows-msvc
# Thread model: posix
# InstalledDir: C:\llvm-build\install\bin
```

## 第8步：配置环境变量（可选）

将以下路径添加到系统PATH：

- `C:\llvm-build\install\bin`
- `C:\llvm-build\install\lib`

## 故障排除

### 常见问题1：内存不足

```bash
# 如果构建时内存不足，可以：
# 1. 减少并行构建作业数
cmake --build . --config Release --parallel 4

# 2. 使用 ninja 生成器（更高效）
# 先安装 ninja，然后在 CMake 配置时使用 -G "Ninja"
```

### 常见问题2：编译错误

- 确保 Windows SDK 版本正确
- 尝试禁用某些组件：

```cmake
-DLLVM_ENABLE_ASSERTIONS=OFF
-DLLVM_ENABLE_EH=ON
-DLLVM_ENABLE_RTTI=ON
```

### 常见问题3：Git 克隆失败

```bash
# 如果网络问题导致克隆失败，可以：
# 1. 使用国内镜像
git clone --depth 1 https://gitee.com/mirrors/llvm-project.git

# 2. 或下载源码包
# 从 https://github.com/llvm/llvm-project/releases/tag/llvmorg-10.0.1
```

## 简化构建脚本

创建一个构建脚本 `build_clang.bat`：

```batch
@echo off
setlocal

set SOURCE_DIR=C:\llvm-build\llvm-project\llvm
set BUILD_DIR=C:\llvm-build\build
set INSTALL_DIR=C:\llvm-build\install

echo Configuring LLVM/Clang...
cmake -G "Visual Studio 15 2017 Win64" ^
  -B "%BUILD_DIR%" ^
  -S "%SOURCE_DIR%" ^
  -DCMAKE_INSTALL_PREFIX="%INSTALL_DIR%" ^
  -DLLVM_ENABLE_PROJECTS="clang" ^
  -DLLVM_TARGETS_TO_BUILD="X86" ^
  -DCMAKE_BUILD_TYPE=Release

if %ERRORLEVEL% neq 0 (
    echo Configuration failed
    pause
    exit /b 1
)

echo Building LLVM/Clang...
cmake --build "%BUILD_DIR%" --config Release --parallel 8

if %ERRORLEVEL% neq 0 (
    echo Build failed
    pause
    exit /b 1
)

echo Installing...
cmake --build "%BUILD_DIR%" --config Release --target INSTALL

echo Done! Clang installed to %INSTALL_DIR%
pause
```

## 构建时间预估

- 8核CPU + 32GB RAM：约2-3小时
- 4核CPU + 16GB RAM：约4-6小时
- 构建过程中会占用大量磁盘空间（约15-20GB）

这个教程应该能帮助您成功构建 Clang 10.0.1。如果遇到具体问题，请提供错误信息以便进一步诊断。
