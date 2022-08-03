
# 在 clang 中 使用 libc++ 库

## 0. 简介

`LLVM` 的 `C++` 库是 `libc++`，`gcc` 的 `C++` 库叫 `libstdc++`，又名 `Glib C++` 。

大多数情况下，即使 `C++` 编译器是 `clang` 时，使用的也可能是 `gcc` 的 `libstdc++` 库，而不是 `libc++` 。

## 1. 安装 libc++ 库

即使您安装了 `clang`，默认也是不带 `libc++` 的，需要另外安装。

`Ubuntu` 中安装 `libc++`：

```bash
sudo apt-get install libc++-dev
sudo apt-get install libc++abi-dev
```

注：安装 `libc++-dev` 的同时，还会安装 `libc++abi1` 和 `libc++1` 。

## 2. 在 CMake 中设置

使用 `clang` 的 `libc++` ：

```text
set(CMAKE_CXX_FLAGS        "${CMAKE_CXX_FLAGS} -std=c++11 -stdlib=libc++")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lc++ -stdlib=libc++")
```

使用 `gcc` 的 `libstdc++` ：

```text
set(CMAKE_CXX_FLAGS        "${CMAKE_CXX_FLAGS} -std=c++11 -stdlib=libstdc++")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lstdc++ -stdlib=libstdc++")
```

## 3. 修改软链接

如果想在 `gcc` 中也使用 `libc++`，可以修改 `/usr/bin/c++` 的软链接：

```bash
ln -sf /usr/bin/clang++-libc++ /usr/bin/c++
```

让其指向 `/usr/bin/clang++-libc++` ：

```bash
ll /usr/bin/c++
/usr/bin/c++ -> /usr/bin/clang++-libc++*
```

## 4. 修改软链接

恢复 `/usr/bin/c++` 原本的软链接：

如果使用过 `update-alternatives`，那么是这样的：

```bash
ll /usr/bin/c++
/usr/bin/c++ -> /etc/alternatives/c++*

# 恢复软链接
ln -sf /etc/alternatives/c++ /usr/bin/c++
```

如果没有使用过 `update-alternatives`，则是这样的：

```bash
ll /etc/alternatives/c++*
/etc/alternatives/c++ -> /usr/bin/g++*

# 恢复软链接
ln -sf /usr/bin/g++ /usr/bin/c++
```

## 5. 参考文章

1. [`Clang doesn't see basic headers`]

    [https://stackoverflow.com/questions/26333823/clang-doesnt-see-basic-headers](https://stackoverflow.com/questions/26333823/clang-doesnt-see-basic-headers)
