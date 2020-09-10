# 如何使用 SSE 4.2 的 PCMPxSTRx 指令

tags: "Intel", "SIMD", "SSE 4.2", "PCMPxSTRx", "PCMPISTRI", "PCMPISTRM", "字符串匹配", "String Match"

## 1. SIMD 简介

现代的 `CPU` 大多都提供了 [`单指令流多数据流`](https://zh.wikipedia.org/wiki/%E5%8D%95%E6%8C%87%E4%BB%A4%E6%B5%81%E5%A4%9A%E6%95%B0%E6%8D%AE%E6%B5%81)（`SIMD`，`Single Instruction Multiple Data`）指令集，最常见的是用于大量的浮点数计算。但其实也可以用在文字处理方面，`Intel` 在 `SSE 4.2` 指令集中就加入了字符串处理的指令，这就是 `PCMPxSTRx` 系列指令。

![Intel Core i7](./images/intel-core-i7.jpg)

（配图为 `2008` 年发售的 `Intel Core i7` 芯片，它采用的 `Nehalem` 是第一个支持 `SSE 4.2` 的微架构。）

## 2. PCMPxSTRx 指令

在 `Intel` 的 `SSE 4.2` 指令集中，有一个 `PCMPxSTRx` 系列指令，它可以一次对一组字符（16个字节）与另一组字符（16个字节）同时作比较，也就是说这一条指令一次可以做最多 "`16 × 16 = 256`" 次的 `字符` 比较。虽然它没有采取任何优化算法，但是由于硬件指令的暴力并行，还是能对字符串匹配、搜索的性能产生巨大的提升。

`PCMPxSTRx` 指令其实是一个指令系列的统称，其中 `x` 代表通配符，它包含了下面的四条具体指令：

PCMP[E](https://baidu.com)STR[I](https://baidu.com)，PCMP[E](https://baidu.com)STR[M](https://baidu.com)，PCMP[I](https://baidu.com)STR[I](https://baidu.com)，PCMP[I](https://baidu.com)STR[M](https://baidu.com)

那么这四条指令分别是什么意思呢？请看下面的表：

|                                                                                    |   返回索引<br/>返回匹配字符串的<br/>索引值到 %rcx   | 返回Mask<br/>返回字符比较结果的<br/>bitmask 到 %xmm0 |
| :--------------------------------------------------------------------------------: | :-------------------------------------------------: | :--------------------------------------------------: |
| 显式的指定字符串的长度，<br/>xmm1 的长度保存在 %rdx，<br/>xmm2 的长度保存在 %rax。 | PCMP[E](https://baidu.com)STR[I](https://baidu.com) | PCMP[E](https://baidu.com)STR[M](https://baidu.com)  |
|                     隐式的字符串长度，<br/>以字符串终止符结束                      | PCMP[I](https://baidu.com)STR[I](https://baidu.com) | PCMP[I](https://baidu.com)STR[M](https://baidu.com)  |


## X. 参考文章

* [1]: [RapidJSON 代码剖析（二）：使用 SSE 4.2 优化字符串扫描](https://zhuanlan.zhihu.com/p/20037058)

* [2]: [Implementing strcmp, strlen, and strstr using SSE 4.2 instructions](https://www.strchr.com/strcmp_and_strlen_using_sse_4.2)

* [3]: [sse 4.2带来的优化](https://www.zzsec.org/2013/08/using-sse_4.2/)

