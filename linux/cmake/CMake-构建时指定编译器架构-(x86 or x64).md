# CMake 构建时指定编译器架构 (x86 or x64)

## 1. Windows

以 `vs2015 x64` 编译器为例，`cmake` 命令如下：

```bash
cmake -G "Visual Studio 14 Win64" path\to\source\dir
```

去掉Win64，就是32bit：

```bash
cmake -G "Visual Studio 14" path\to\source\dir
```

另外一种等价方式，用命令行参数-A来指定架构（x64或者ARM）：

```bash
cmake -A x64 path\to\source\dir
```

更多参考：

https://cmake.org/cmake/help/v3.1/manual/cmake-generators.7.html#ide-build-tool-generators

`Windows` 下如果用了 `cmake -G "Visual Studio 14"` 命令，则 `cmake` 会给你生成 `Visual Studio` 工程文件相关的文件，比如：`Project.sln`，这时要编译生成必须用 `msbuild` 命令，比如：

```bash
msbuild Project.sln
```

完整步骤是：

```bash
cmake -G "Visual Studio 14 Win64" path\to\source\dir

msbuild Project.sln 
```

**Windows nmake**

`Windows` 还提供了一种构建命令：`nmake`。使用命令如下：

```bash
cmake -G "NMake Makefiles" path\to\source\dir

nmake
```

如果要为 `nmake` 指定 `x64/x86`，还不清楚 `cmake` 有没相关参数设置，目前我知道的方法如下，

以 `vs2015` 为例，打开 `cmd`，定位到 `\Microsoft Visual Studio 14.0\VC\` 目录下，然后执行命令：

```bash
vcvarsall.bat x64 
```

如果要 `32` 位，就执行：

```bash
vcvarsall.bat x86
```

执行完后再跳转到要构建的工程目录下，接着执行：

```bash
cmake -G "NMake Makefiles" path\to\source\dir
nmake
```

这样生成出来的程序就是 `x86` 或者 `x64` 版本。

## 2. Linux

设置 `CFLAGS`（或者`CXXFLAGS`）为 `-m32` 或者 `-m64`，例如：

```bash
export CFLAGS=-m32
export CXXFLAGS=-m32
```

## 3. Mac OSX

32 bit

```bash
cmake -DCMAKE_OSX_ARCHITECTURES=i386 /path/to/source/dir
```

64 bit

```bash
cmake -DCMAKE_OSX_ARCHITECTURES=x86_64 /path/to/source/dir will compile 
```

96-bit universal

```bash
cmake"-DCMAKE_OSX_ARCHITECTURES=x86_64;i386"/path/to/source/dir
```

## 4. 参考文章

* 【1】: [http://stackoverflow.com/questions/5334095/cmake-multiarchitecture-compilation]()
* 【2】: [https://www.iteye.com/blog/aigo-2294970]()
