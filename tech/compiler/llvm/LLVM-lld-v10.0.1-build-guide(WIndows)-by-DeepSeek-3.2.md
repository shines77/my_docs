# LLVM lld v10.0.1 构建指南 (Windows)

我将为您提供在 Windows 上使用 CMake 和 Visual Studio 2017 构建 LLD 10.0.1（LLVM链接器）的详细教程。LLD 是 LLVM 项目的一部分，可以与 Clang 一起构建。

## 系统要求

- Windows 10 或更高版本
- Visual Studio 2017（15.9或更高版本）
- 至少20GB可用磁盘空间
- 至少8GB RAM（建议16GB）
- Python 3.6+
- Git
- CMake 3.13+

## 第1步：环境准备

### 1.1 安装必要软件（如果尚未安装）

```powershell
# 检查已安装软件
cmake --version
git --version
python --version

# 如果没有安装，请先安装：
# 1. Visual Studio 2017 with C++ tools
# 2. CMake 3.13+ 
# 3. Python 3.6+
# 4. Git for Windows
```

## 第2步：获取源代码

```bash
# 打开命令提示符（管理员权限）
mkdir C:\llvm-build
cd C:\llvm-build

# 方法1：克隆完整 LLVM项目（推荐）
git clone --depth 1 --branch llvmorg-10.0.1 https://github.com/llvm/llvm-project.git

# 方法2：如果只需要 LLD
git clone --depth 1 --branch llvmorg-10.0.1 https://github.com/llvm/llvm-project.git
# 或者只克隆LLD相关部分
```

## 第3步：创建目录结构

```bash
cd C:\llvm-build

# 创建构建和安装目录
mkdir build
mkdir install

# 最终目录结构：

# C:\llvm-build\
#   ├── llvm-project\lld    # 源代码
#   ├── lld-build\          # 构建目录
#   └── lld-install\        # 安装目录
```

## 第4步：配置CMake构建

### 4.1 使用CMake GUI配置（推荐）

1. 打开CMake GUI
2. 设置源代码路径：`C:/llvm-build/llvm-project/lld`
3. 设置构建路径：`C:/llvm-build/lld-build`
4. 点击"Configure"

5. 选择生成器：

   - `Visual Studio 15 2017`

6. 配置以下选项：

| 变量名 | 值 | 说明 |
|--------|-----|------|
| **CMAKE_INSTALL_PREFIX** | C:/llvm-build/lld-install | 安装路径 |
| **LLVM_ENABLE_PROJECTS** | lld | 启用LLD项目 |
| **LLVM_TARGETS_TO_BUILD** | X86 | 只构建x86目标 |
| **CMAKE_BUILD_TYPE** | Release | 发布版本 |
| **LLVM_OPTIMIZED_TABLEGEN** | ON | 优化TableGen |
| **LLVM_INCLUDE_TESTS** | OFF | 不包含测试（加快构建） |
| **LLVM_INCLUDE_EXAMPLES** | OFF | 不包含示例 |
| **LLVM_USE_CRT_RELEASE** | MT | 使用静态运行时 |
| **BUILD_SHARED_LIBS** | OFF | 构建静态库 |
| **LLVM_ENABLE_ASSERTIONS** | OFF | 禁用断言（发布版） |
| **LLVM_USE_NEW_LIB_ZLIB** | ON | 使用内置zlib |

7. 点击"Configure"直到无红色错误
8. 点击"Generate"

### 4.2 使用命令行配置

```batch
cd C:\llvm-build\lld-build

# 基本配置（构建LLD）
cmake ../llvm-project/lld `
  -G "Visual Studio 15 2017" -A x64 -Thost=x64 `
  -DCMAKE_INSTALL_PREFIX="C:/llvm-build/lld-install" `
  -DLLVM_ENABLE_PROJECTS="lld" `
  -DLLVM_TARGETS_TO_BUILD="X86" `
  -DCMAKE_BUILD_TYPE=Release `
  -DLLVM_OPTIMIZED_TABLEGEN=ON `
  -DLLVM_INCLUDE_TESTS=OFF `
  -DLLVM_USE_CRT_RELEASE=MT `
  -DBUILD_SHARED_LIBS=OFF `
  -DLLVM_USE_NEW_LIB_ZLIB=ON `
  -DLLVM_ENABLE_EH=ON `
  -DLLVM_ENABLE_RTTI=ON `
  -DLLVM_EXPORT_SYMBOLS_FOR_PLUGINS=ON `
  -DCMAKE_POLICY_VERSION_MINIMUM="3.5" `
  -DCMAKE_POLICY_DEFAULT_CMP0000=NEW `
  -DCMAKE_POLICY_DEFAULT_CMP0053=NEW `
  -Wno-dev
```

### 4.3 如果需要构建Clang + LLD（完整工具链）

```batch
cd C:\llvm-build\lld-build

# Clang 和 LLD 一起构建
cmake ../llvm-project/lld `
  -G "Visual Studio 15 2017" -A x64 -Thost=x64 `
  -DCMAKE_INSTALL_PREFIX="C:/llvm-build/lld-install" `
  -DLLVM_ENABLE_PROJECTS="clang;lld" `
  -DLLVM_TARGETS_TO_BUILD="X86" `
  -DCMAKE_BUILD_TYPE=Release `
  -DLLVM_OPTIMIZED_TABLEGEN=ON `
  -DLLVM_INCLUDE_TESTS=OFF `
  -DLLVM_USE_CRT_RELEASE=MT `
  -DBUILD_SHARED_LIBS=OFF `
  -DLLVM_USE_NEW_LIB_ZLIB=ON `
  -DLLVM_ENABLE_EH=ON `
  -DLLVM_ENABLE_RTTI=ON `
  -DLLVM_EXPORT_SYMBOLS_FOR_PLUGINS=ON `
  -DCLANG_DEFAULT_LINKER="lld" `
  -DCLANG_DEFAULT_RTLIB="compiler-rt" `
  -DCLANG_DEFAULT_UNWINDLIB="libunwind" `
  -DLLVM_ENABLE_LLD=ON `
  -DCMAKE_POLICY_VERSION_MINIMUM="3.5" `
  -DCMAKE_POLICY_DEFAULT_CMP0000=NEW `
  -DCMAKE_POLICY_DEFAULT_CMP0053=NEW `
  -Wno-dev
```

## 第5步：构建LLD

### 5.1 使用Visual Studio构建

```bash
# 打开生成的解决方案
"C:\Program Files (x86)\Microsoft Visual Studio\2017\Professional\Common7\IDE\devenv.exe" C:\llvm-build\build\LLVM.sln
```

在Visual Studio中：

1. 在工具栏选择"Release"配置
2. 右键点击"ALL_BUILD" -> "生成"
3. 或只构建lld目标：
   - 在解决方案资源管理器中找到"lld"项目
   - 右键 -> "生成"

### 5.2 使用命令行构建

```batch
cd C:\llvm-build\lld-build

# 方法1：构建所有（包括依赖）
cmake --build . --config Release --target ALL_BUILD -- /m /p:Platform=x64

# 方法2：只构建lld
cmake --build . --config Release --target lld

# 方法3：并行构建（推荐）
cmake --build . --config Release --parallel 8

# 方法4：构建特定目标
cmake --build . --config Release --target lld-link  # Windows链接器
cmake --build . --config Release --target lld       # 所有LLD工具
```

## 第6步：安装LLD

```batch
cd C:\llvm-build\lld-build

# 安装所有
cmake --build . --config Release --target INSTALL

# 或只安装LLD相关文件
cmake --build . --config Release --target install-lld
```

## 第7步：验证安装

```batch
# 检查安装目录
dir C:\llvm-build\lld-install\bin\

# 测试lld-link（Windows的LLD链接器）
C:\llvm-build\lld-install\bin\lld-link --version

# 测试其他LLD组件
C:\llvm-build\lld-install\bin\ld.lld --version    # ELF链接器
C:\llvm-build\lld-install\bin\lld --version       # 通用驱动程序
C:\llvm-build\lld-install\bin\wasm-ld --version   # WebAssembly链接器

# 应该显示类似：
# LLD 10.0.1
```

## 第8步：配置环境

### 8.1 添加环境变量

```batch
# 将以下路径添加到系统PATH环境变量：
# C:\llvm-build\lld-install\bin
# C:\llvm-build\lld-install\lib

# 临时设置（当前会话）
set PATH=C:\llvm-build\lld-install\bin;%PATH%
```

### 8.2 测试使用LLD

```batch
# 1. 创建测试文件
echo int main() { return 0; } > test.c

# 2. 使用Clang + LLD编译（如果有Clang）
C:\llvm-build\lld-install\bin\clang test.c -fuse-ld=lld -o test.exe

# 3. 直接使用lld-link
C:\llvm-build\lld-install\bin\clang-cl test.c /link /lld:lld-link

# 4. 检查生成的可执行文件
test.exe
echo %ERRORLEVEL%
```

## 第9步：构建脚本示例

创建 `build_lld.bat`：

```batch
@echo off
setlocal enabledelayedexpansion

echo ========================================
echo LLD 10.0.1 Windows Build Script
echo ========================================

set BUILD_ROOT=C:\llvm-build
set SOURCE_DIR=%BUILD_ROOT%\llvm-project\llvm
set BUILD_DIR=%BUILD_ROOT%\lld-build
set INSTALL_DIR=%BUILD_ROOT%\lld-install
set GENERATOR="Visual Studio 15 2017"
set PARALLEL_JOBS=8

echo Step 1: Cleaning previous build...
if exist "%BUILD_DIR%" rmdir /s /q "%BUILD_DIR%"
if exist "%INSTALL_DIR%" rmdir /s /q "%INSTALL_DIR%"
mkdir "%BUILD_DIR%"
mkdir "%INSTALL_DIR%"

echo Step 2: Configuring with CMake...
cd "%BUILD_DIR%"
cmake %SOURCE_DIR% `
  -G %GENERATOR% `
  -DCMAKE_INSTALL_PREFIX="%INSTALL_DIR%" `
  -DLLVM_ENABLE_PROJECTS="lld" `
  -DLLVM_TARGETS_TO_BUILD="X86" `
  -DCMAKE_BUILD_TYPE=Release `
  -DLLVM_OPTIMIZED_TABLEGEN=ON `
  -DLLVM_INCLUDE_TESTS=OFF `
  -DLLVM_INCLUDE_EXAMPLES=OFF `
  -DLLVM_USE_CRT_RELEASE=MT `
  -DBUILD_SHARED_LIBS=OFF `
  -DLLVM_USE_NEW_LIB_ZLIB=ON `
  -DLLVM_ENABLE_ASSERTIONS=OFF

if %ERRORLEVEL% neq 0 (
    echo ERROR: CMake configuration failed!
    pause
    exit /b 1
)

echo Step 3: Building LLD...
cmake --build . --config Release --parallel %PARALLEL_JOBS%

if %ERRORLEVEL% neq 0 (
    echo ERROR: Build failed!
    pause
    exit /b 1
)

echo Step 4: Installing...
cmake --build . --config Release --target INSTALL

if %ERRORLEVEL% neq 0 (
    echo ERROR: Installation failed!
    pause
    exit /b 1
)

echo Step 5: Verifying installation...
if exist "%INSTALL_DIR%\bin\lld-link.exe" (
    "%INSTALL_DIR%\bin\lld-link.exe" --version
    echo.
    echo ========================================
    echo SUCCESS: LLD has been built and installed!
    echo Installation directory: %INSTALL_DIR%
    echo ========================================
) else (
    echo ERROR: lld-link.exe not found!
)

pause
```

## 故障排除

### 问题1：CMake配置错误

```batch
# 错误：找不到合适的生成器
# 解决方案：确保VS 2017安装正确
"C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\Common7\Tools\VsDevCmd.bat"
```

### 问题2：构建失败（内存不足）

```batch
# 减少并行作业数
cmake --build . --config Release --parallel 4

# 或使用ninja生成器（更高效）
cmake -G "Ninja" ...
```

### 问题3：链接错误

```batch
# 尝试禁用某些特性
-DLLVM_ENABLE_THREADS=OFF
-DLLVM_ENABLE_PIC=OFF
```

### 问题4：下载源代码慢

```batch
# 使用国内镜像
git clone https://gitee.com/mirrors/llvm-project.git
cd llvm-project
git checkout llvmorg-10.0.1
```

## LLD组件说明

构建完成后，您会得到以下工具：

- `lld-link.exe` - Windows COFF链接器（替代link.exe）
- `ld.lld.exe` - ELF链接器（Linux/Unix格式）
- `wasm-ld.exe` - WebAssembly链接器
- `lld.exe` - 通用驱动程序

## 构建时间预估

- 仅LLD：30-60分钟
- LLD + Clang：2-4小时
- 完整LLVM：4-8小时

## 使用LLD的Visual Studio项目配置

在VS 2017中使用lld-link：

1. 项目属性 -> 链接器 -> 常规

   - 启用 "Use LLD Linker" = Yes

2. 或使用命令行选项：

   ```
   /link /lld:lld-link
   ```

这个教程应该能帮助您成功构建LLD 10.0.1。如果您遇到特定错误，请提供错误信息以便进一步诊断。
