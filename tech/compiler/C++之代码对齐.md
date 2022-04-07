# C++ 之 代码对齐

## 1. clang & LLVM

1. [LLVM 编译器学习笔记之二十一 -- Label 对齐](https://blog.csdn.net/zhongyunde/article/details/117751859)

    1. 循环边界对齐
    2. Code alignment options in llvm
    3. Branch Alignment
    4. ICC compiler try to align blocks of data to cacheline size

2. [Code alignment options in llvm](https://easyperf.net/blog/2018/01/25/Code_alignment_options_in_llvm)

    Contents:

    * align-all-functions
    * align-all-blocks
    * align-all-nofallthru-blocks

    ```shell
    -mllvm -align-all-functions=5          可以，没问题，默认应该也是 32 字节
    -mllvm -align-all-blocks=5             这个最实用，但是好像也不完全是我们想要的
    -mllvm -align-all-nofallthru-blocks=5  这个有一点用
    ```

3. [Code alignment issues.](https://easyperf.net/blog/2018/01/18/Code_alignment_issues)

    Compare the "`-mllvm -align-all-functions=5`" and "`-mllvm -align-all-blocks=5`" result.

## 2. Intel C++ compiler

1. [Intel C++ Compiler: #pragma code_align(n)](https://www.intel.com/content/www/us/en/develop/documentation/cpp-compiler-developer-guide-and-reference/top/compiler-reference/pragmas/intel-specific-pragma-reference/code-align-1.html)

    code_align

    * Specifies the byte alignment for a loop

    Syntax

    * #pragma code_align(n)

    Arguments

    * n

        Optional. A positive integer initialization expression indicating the number of bytes for the minimum alignment boundary. Its value must be a power of 2, between 1 and 4096, such as 1, 2, 4, 8, and so on.

        If you specify 1 for n, no alignment is performed. If you do not specify n, the default alignment is 16 bytes.

## 3. GCC

See: [https://stackoverflow.com/questions/1912833/c-function-alignment-in-gcc](https://stackoverflow.com/questions/1912833/c-function-alignment-in-gcc)

1. align-functions

    ```cpp
    #pragma GCC push_options
    #pragma GCC optimize ("align-functions=16")
    #pragma GCC optimize ("unroll-loops")
    ........
    #pragma GCC pop_options
    ```

2. [GCC's attribute syntax](http://gcc.gnu.org/onlinedocs/gcc/Attribute-Syntax.html)

    ```cpp
    //add 5 to each element of the int array.
    __attribute__((optimize("align-functions=16")))
    __attribute__((optimize("unroll-loops")))
    void add5(int a[20]) {
        int i = 19;
        for(; i > 0; i--) {
            a[i] += 5;
        }
    }
    ```

3. `-falign-functions=16`

    ```shell
    gcc -O2 -Wall -falign-functions=16
    ```

4. GCC Optimize Options

    [GCC Optimize Options](https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html#Optimize-Options)

    ```cpp
    gcc -falign-labels=n
    gcc -falign-loops=n
    gcc -falign-jumps=n
    ```

    或者

    ```cpp
    #if defined(__GNUC__) && !defined(__clang__)
    #pragma GCC push_options
    #pragma GCC optimize ("align-labels=n")
    #pragma GCC optimize ("align-loops=n")
    #pragma GCC optimize ("align-jumps=n")
    #endif

    ........

    #if defined(__GNUC__) && !defined(__clang__)
    #pragma GCC pop_options
    #endif
    ```

## X. Other

* GCC

    ```shell
    gcc -O2 -march=native -mtune=generic
    ```

    People need a note: Use `-march` if you ONLY run it on your processor, use `-mtune` if you want it safe for other processors.

    There's also `-mtune=generic`. "generic" makes GCC produce code that runs best on current CPUs (meaning of generic changes from one version of GCC to another).

  * -mtune=name

      This option is very similar to the "-mcpu=" option, except that
      instead of specifying the actual target processor type, and hence
      restricting which instructions can be used, it specifies that GCC
      should tune the performance of the code as if the target were of
      the type specified in this option, but still choosing the instruc-
      tions that it will generate based on the cpu specified by a "-mcpu="
      option.

  `march` 指定的是当前 cpu 的架构，而 `mtune` 是真正作用于某一型号 cpu 的选项。`march` 是 “紧约束”，`mtune` 是 “松约束”，`mtune` 可以提供后向兼容性。

* POD (Plain Old Data) type
