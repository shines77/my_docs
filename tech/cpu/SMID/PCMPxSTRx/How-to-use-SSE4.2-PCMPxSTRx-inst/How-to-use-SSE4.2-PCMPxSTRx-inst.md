# 如何使用 SSE 4.2 的 PCMPxSTRx 指令

tags: "Intel", "SIMD", "SSE 4.2", "PCMPxSTRx", "PCMPISTRI", "PCMPISTRM", "字符串匹配", "String Match"

## 1. SIMD 简介

现代的 `CPU` 大多都提供了 [`单指令流多数据流`](https://zh.wikipedia.org/wiki/%E5%8D%95%E6%8C%87%E4%BB%A4%E6%B5%81%E5%A4%9A%E6%95%B0%E6%8D%AE%E6%B5%81)（`SIMD`，`Single Instruction Multiple Data`）指令集，最常见的是用于大量的浮点数计算。但其实也可以用在文字处理方面，`Intel` 在 `SSE 4.2` 指令集中就加入了字符串处理的指令，这就是 `PCMPxSTRx` 系列指令。

这里简单的介绍一下 `x86` 架构下的 `SIMD`，在 `SSE 4.2` 指令集之前，`Intel` 和 `AMD` 共同维护和开发了 `MMX`，`SSE`，`SSE 2`，`SSE 3`，`SSE 4`，`SSE 4.1`，`SSE 4.a`，`3D Now` 等指令集。在 `SSE 4.2` 指令集之后，最新的还有 `SSE 5`，`AVX`，`AVX 2`，`FMA`，`AVX 512` 等等指令集。

![Intel Core i7](./images/intel-core-i7.jpg)

（配图为 `2008` 年发售的 `Intel Core i7` 芯片，它采用的 `Nehalem` 是第一个支持 `SSE 4.2` 的微架构。）

## 2. SSE 4.2 指令集

`SSE 4.2` 指令集都包含了哪些指令？

这里也简单介绍一下，通过 `Intel` 官方网站的指令指南：[Intel: Intrinsics Guide: SSE 4.2](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=914&techs=SSE4_2)，可以看到。

`SSE 4.2` 指令集主要包含了三类指令：

* 用于字符或文字处理的 `PCMPxSTRx` 系列指令
* 用于校验或者哈希的 `CRC32` 系列指令
* 用于 packed 的 `64位` 数据比较的 `_mm_cmpgt_epi64()` 指令（这个是填以前的坑的，因为 `SSE 4.2` 之前没实现）

忽略第 `3` 类那条填旧坑的指令（而且也只有一条而已），`SSE 4.2` 其实只有 `PCMPxSTRx` 和 `CRC32` 两大类，其中绝大部分都是 `PCMPxSTRx` 指令。所以，如果你认为 `PCMPxSTRx` 就是 `SSE 4.2` 指令集的代表，也不为过。

下图中，以 `_mm_cmpestr` 和 `_mm_cmpistr` 开头的函数都是 `PCMPxSTRx` 指令。

（注：为了编程方便，`Intel` 把指令包装成函数，这些函数“一一对应”着某一条具体的 `CPU` 指令。）

![Intel Core i7](./images/Intel-SSE-4.2-insts.png)

## 3. PCMPxSTRx 指令

### 3.1 原理

`PCMPxSTRx` 系列指令有着很强的 `并行比较` 能力，也许是 `x86` 指令集中最复杂的指令之一。

那么，`PCMPxSTRx` 指令到底能干什么？工作原理？

它可以一次对一组字符（16个Bytes或8个Word）与另一组字符（16个Bytes或8个Word）同时作比较，也就是说这一条指令一次可以最多做 "`16 × 16 = 256`" 次的 `字符` 比较。虽然它没有采取任何优化算法，但是由于硬件指令的暴力并行，还是能对字符串匹配、搜索和比较的性能产生巨大的提升。

我们以最常用、最有价值的 `Equal Ordered`（顺序相等）模式为例，大致的工作原理如下所示：

![pcmpistri xmm1, xmm2, 0x0C](./images/pcmpistri_equal_ordered_mode.png)

上图所示指令的 `C/C++` 伪代码如下：

```c
char * operand1 = "We";
char * operand2 = "WhenWeWillBeWed!"

// imm8 = 0x0C
uint8_t imm8 = _SIDD_CHAR_OPS | _SIDD_CMP_EQUAL_ORDERED
    | _SIDD_POSITIVE_POLARITY | _SIDD_LEAST_SIGNIFICANT;

// pcmpistri  xmm1, xmm2, 0x0C
int index = _mm_cmpistri(operand1, operand2, imm8);
```

* `Equal Ordered` = 0x0C，imm[3:2] = 11b，判断 `operand1` 是否是 `operand2` 的子串。

```java
operand2 = "WhenWeWillBeWed!"
operand1 = "We"
IntRes1  =  0000100000001000
```

我们可以看到 `"WhenWeWillBeWed!"` 中包含了 `"We"` 子串两次，分别是在索引 `4` 和 `12` 的位置（`IntRes1` 从左往右数）。

笔者注：`IntRes1` 其实不是一个 `bit` 数组，而是一个 `Byte` 数组，上图和上面的代码中的 `IntRes1` 的 `0` 和 `1`，它其实不是一个 `bit`，而是一个 `Byte`，`1` (0x01) 其实是 `11111111` (0xFF)，只是为了作图和表述方便，简写成一个 bit 的 `1` 。

### 3.2 指令详解

`PCMPxSTRx` 指令其实是一个指令系列的统称，其中 `x` 代表通配符，它包含了下面的四条具体指令：

* pcmp[**e**](https://baidu.com)str[**i**](https://baidu.com)

* pcmp[**e**](https://baidu.com)str[**m**](https://baidu.com)

* pcmp[**i**](https://baidu.com)str[**i**](https://baidu.com)

* pcmp[**i**](https://baidu.com)str[**m**](https://baidu.com)

那么这四条指令分别代表什么意思呢？请看下面的表：

|                                                                                    |      返回索引<br/>(返回匹配字符串的<br/>索引值到 %rcx)      |   返回Mask<br/>(返回字符比较结果的<br/>bitmask 到 %xmm0)    |
| :--------------------------------------------------------------------------------: | :---------------------------------------------------------: | :---------------------------------------------------------: |
| 显式的指定字符串的长度，<br/>xmm1 的长度保存在 %rdx，<br/>xmm2 的长度保存在 %rax。 | pcmp[**e**](https://baidu.com)str[**i**](https://baidu.com) | pcmp[**e**](https://baidu.com)str[**m**](https://baidu.com) |
|                     隐式的字符串长度，<br/>以字符串终止符结束                      | pcmp[**i**](https://baidu.com)str[**i**](https://baidu.com) | pcmp[**i**](https://baidu.com)str[**m**](https://baidu.com) |

为了方便理解，先来了解一下 `PCMPxSTRx` 指令的一般格式，例如：

```asm
pcmpistri  %xmm1, %xmm2, imm8
```

注：其中 `%xmm1`, `%xmm2` 代表任意 `SSE` 的 `128位` 寄存器（`SSE` 一共有 `xmm0` ~ `xmm15` 16 个寄存器），`imm8` 是一个 `8bit` 的立即数（常量），用于配置 `pcmpistri` 指令，定义指令具体的执行功能，稍后会详细介绍。常用的值是：imm8 = 0x0C（十六进制）。

从上表可以看出，在 `PCMPxSTRx` 指令中，

前面的通配符：

* 如果是 `e` 的话，表示显示的指定输入的字符串的长度，`xmm1` 的长度保存在 `%rdx` 寄存器，`xmm2` 的长度保存在 `%rax` 寄存器。

* 如果是 `i`，则表示隐式的字符串长度，输入的字符串以终止符 “`\0`” 结束。

后面的通配符：

* 如果是 `i`，则表示返回的结果是索引值，即 `IntRes1` 中（二进制）从最高位开始数 (MSB) 或最低位开始数 (LSB) 第一个为 `1` 的索引位置，结果存到 `%rcx` 寄存器。

* 如果是 `m`，则表示返回的结果是一个 `BitMask` (Bit位掩码)，且这个值保存到 `%xmm0` 寄存器中（这里的 `xmm0` 是真实的寄存器名，也就是说会占用 `SSE` 的 `xmm0` 寄存器）。

`PCMPxSTRx` 指令也支持 32 位系统，此时，前面提到的 `%rax`，`%rdx`，`%rcx` 寄存器相对应的是 `%eax`，`%edx`，`%ecx` 寄存器。

所以，我们来总结一下这四条指令的具体含义：

* `pcmpestri`：**P**acked **Com**pare **E**xplicit Length **Str**ings, Return **I**ndex。<br/>-------------（批量比较显式指定长度的字符串，返回索引值）

* `pcmpestrm`：**P**acked **Com**pare **E**xplicit Length **Str**ings, Return **M**ask。<br/>-------------（批量比较显式指定长度的字符串，返回掩码）

* `pcmpistri`：**P**acked **Com**pare **I**mplicit Length **Str**ings, Return **I**ndex。<br/>-------------（批量比较隐式长度的字符串，返回索引值）

* `pcmpistrm`：**P**acked **Com**pare **I**mplicit Length **Str**ings, Return **M**ask。<br/>-------------（批量比较隐式长度的字符串，返回掩码）

## X. 参考文章

* [1]: [RapidJSON 代码剖析（二）：使用 SSE 4.2 优化字符串扫描](https://zhuanlan.zhihu.com/p/20037058)

* [2]: [Implementing strcmp, strlen, and strstr using SSE 4.2 instructions](https://www.strchr.com/strcmp_and_strlen_using_sse_4.2)

* [3]: [sse 4.2带来的优化](https://www.zzsec.org/2013/08/using-sse_4.2/)

* [4]: [Intel: Intrinsics Guide](https://software.intel.com/sites/landingpage/IntrinsicsGuide/)

* [5]: [PCMPISTRI](https://www.felixcloutier.com/x86/pcmpistri)
