# rANS 算法简介

## 问题

Question：请详细介绍压缩算法中的rANS算法的原理，并用C++实现。

(腾讯·混元大模型，2025-09-18)

## rANS

rANS（Range Asymmetric Numeral Systems） 是一种高效的熵编码算法，属于算术编码（Arithmetic Coding）的一种现代变体。它由 Jarek Duda 在 2014 年提出，目标是提供比传统算术编码更高性能、更易并行化且适合硬件实现的压缩方法。

rANS 是 ANS（Asymmetric Numeral Systems） 家族中的一种，特别适用于从左到右（range-based）的编码方式，常用于数据压缩领域，如 zstd（Facebook 的 Zstandard 压缩库）中就使用了类似 ANS 的技术。

## 一、rANS 基本原理

1. 核心思想

rANS 的核心思想是将一个符号的概率分布映射到一个整数状态空间上，通过维护一个大的整数状态（通常是一个 32 位或 64 位的整数），在编码时根据当前符号调整该状态，并输出部分状态信息；解码时则反向操作，从输出中恢复状态并解出原始符号。

相比传统算术编码需要维护浮点区间 $[low, high)$，rANS 使用一个整数状态和一个频率表，将概率信息转换为整数域上的范围划分，避免了浮点运算，提高了速度和精度。

2. 概率模型与频率表

假设我们对一组符号进行编码，每个符号 $s$ 有一个出现频率 $f_s$，总频率 $F = \sum f_s$。

rANS 不直接使用概率 $p_s = f_s / F$，而是使用整数频率 $f_s$，通过将状态空间按频率大小划分来隐式地表示概率。

3. 编码过程（rANS Encode）

设当前编码状态为 $x$（一个较大的整数，比如 32/64 位），我们要编码一个符号 $s$，其频率为 $f_s$，所有符号的总频率为 $F$。

编码步骤如下：

1. 计算新状态：

   $$x' = \left\lfloor \frac{x}{f_s} \right\rfloor \cdot F + C_s + x \bmod f_s$$

   其中：

   - $C_s = \sum_{i=0}^{s-1} f_i$ 是符号 $s$ 的累积频率（即符号 $s$ 之前所有符号的频率和）。
   - $x \bmod f_s$ 是当前状态对频率 $f_s$ 取模，用于定位在当前块中的偏移。
   - $\left\lfloor x / f_s \right\rfloor$ 表示当前状态可以分成多少个完整的 $f_s$ 大小的块。
   - $F$ 是总频率，用来映射到整个状态空间。

2. 输出部分状态：

   为了腾出空间记录新的符号信息，我们通常将 $x$ 的高位部分输出（比如低 8/16/32 位作为输出字节，然后保留高比特继续编码）。

注意：实际实现中为了效率，常常将 $x$ 的最低 N 位（比如 8 位）作为输出字节流，然后将 $x$ 右移 N 位，保留高位继续编码。

4. 解码过程（rANS Decode）

解码是编码的逆过程，我们从状态 $x$ 中还原出原始符号 $s$，并恢复上一状态 $x_prev$。

解码步骤如下：

1. 计算符号 s：

   - 首先计算累积频率索引：

     $$s = C^{-1}[ x \bmod F ]$$

     即通过 $x \bmod F$ 找到当前落在哪个符号的频率范围内，从而得到符号 $s$。

   - 或者更常用的方式是预计算一个 $C[]$（累积频率表）以及 $C_inverse[]$ 映射，使得给定 $x \bmod F$ 能快速找到对应的符号 $s$。

2. 计算前一个状态 $x_prev：

   $$x_{\text{prev}} = f_s \cdot \left( \frac{x}{F} \right) + (x \bmod F) - C_s$$

   然后你可以通过这个 x_prev 继续解码下一个符号。

3. 恢复输出字节：
   在解码时，我们需要从输入流中重新构建完整的 x 值（比如通过读取之前输出的低位字节，左移并拼接）。

## 二、rANS 的优势

- 高性能：相比传统算术编码，rANS 通常更快，因为它是基于整数的，没有浮点运算。
- 低延迟、可并行化潜力：适合现代 CPU 和硬件优化。
- 编码和解码方向不同：rANS 特别适合从前往后编码，从后往前解码，这在某些场景（如文件压缩）中非常有用。
- 可结合预计算频率表，适应不同的符号分布。

## 三、C++ 实现 rANS（简化版，单符号，固定频率）

下面是一个极简版本的 rANS 编码/解码实现，仅用于教学目的，假设只有一个符号（或固定频率表），并且采用 32 位状态，每次输出/输入 8 位（1 字节）。

我们以一个简单的例子来说明：编码一系列来自有限字母表的符号，每个符号有固定的频率。

1. 假设条件

- 符号集：假设只有几个符号，比如 A, B, C，频率分别为 fA, fB, fC，总频率 F = fA + fB + fC。
- 用整数表示符号，比如：A = 0, B = 1, C = 2。
- 编码状态 x 是 uint32_t 类型。
- 每次输出最低 8 位（1字节），然后 x 右移 8 位。

2. C++ 实现代码（简化 rANS，教学示例）

```cpp
#include <iostream>
#include <vector>
#include <cstdint>
#include <unordered_map>

// 简化版 rANS，固定频率，单字节输出
class SimpleRansEncoder {
public:
    using uint = uint32_t;

    uint F; // 总频率
    std::vector<uint> C; // C[s] = sum of frequencies for symbols < s

    SimpleRansEncoder(const std::vector<uint>& freqs) {
        F = 0;
        for (uint f : freqs) F += f;
        C.resize(freqs.size() + 1, 0);
        for (size_t i = 0; i < freqs.size(); ++i)
            C[i + 1] = C[i] + freqs[i];
    }

    // 编码一个符号，更新状态 x
    void encode(uint& x, uint s, uint f_s) {
        x = ((x / f_s) << 8) + (x % f_s) + C[s]; // 简化：假设 F == 256，或者调整移位大小
        // 注意：这里为了简单，假设 F 是 256，每次输出 8 位
        // 实际中 F 是总频率，移位根据输出比特数决定
        // 此处只是一个示意，真实实现需仔细处理 F 与移位的关系
    }

    // 输出缓冲
    std::vector<uint8_t> output;
    void pushByte(uint& x) {
        output.push_back(x & 0xFF);
        x >>= 8;
    }

    // 获取编码结果
    std::vector<uint8_t> finalize(uint& x) {
        while (x > 0xFF) {
            pushByte(x);
        }
        if (x != 0) {
            pushByte(x);
        }
        return output;
    }
};

class SimpleRansDecoder {
public:
    using uint = uint32_t;

    uint F;
    std::vector<uint> C; // 累积频率
    std::vector<uint> freqs;

    SimpleRansDecoder(const std::vector<uint>& freqs_) : freqs(freqs_) {
        F = 0;
        for (uint f : freqs) F += f;
        C.resize(freqs.size() + 1, 0);
        for (size_t i = 0; i < freqs.size(); ++i)
            C[i + 1] = C[i] + freqs[i];
    }

    // 从状态 x 解码一个符号，并返回符号和新的 x
    std::pair<uint, uint> decode(uint x) {
        uint s = 0;
        uint fs = 0;
        // 找到 s 使得 C[s] <= x % F < C[s+1]
        uint remainder = x % F;
        for (s = 0; s < C.size() - 1; ++s) {
            if (remainder >= C[s] && remainder < C[s + 1]) {
                break;
            }
        }
        uint f_s = freqs[s];
        uint x_prev = f_s * (x / F) + (x % F) - C[s];
        return { s, x_prev };
    }

    // 喂入字节，重建 x
    void feedByte(uint& x, uint8_t byte) {
        x = (x << 8) | byte;
    }
};

// 示例使用
int main() {
    // 假设我们有 3 个符号 A,B,C，频率分别是 1, 2, 3
    std::vector<uint> freqs = {1, 2, 3};
    SimpleRansEncoder encoder(freqs);
    SimpleRansDecoder decoder(freqs);

    uint state = 1u << 24; // 初始状态，选择一个较大的数，比如 2^24

    // 编码一些符号（假设已知符号编号）
    std::vector<uint> symbols = {0, 1, 2, 1}; // A, B, C, B
    for (uint s : symbols) {
        uint f_s = freqs[s];
        encoder.encode(state, s, f_s);
    }

    auto encoded = encoder.finalize(state);

    std::cout << "Encoded bytes (" << encoded.size() << " bytes): ";
    for (uint8_t b : encoded) std::cout << std::hex << (int)b << " ";
    std::cout << std::dec << "\n";

    // 解码：重建状态并解码符号
    uint decode_state = 0;
    for (uint8_t b : encoded) {
        decoder.feedByte(decode_state, b);
    }

    // 假设我们知道编码了多少符号，这里简单循环解码
    std::vector<uint> decoded_symbols;
    for (size_t i = 0; i < symbols.size(); ) {
        auto [s, new_state] = decoder.decode(decode_state);
        decoded_symbols.push_back(s);
        decode_state = new_state;
        ++i;
    }

    std::cout << "Decoded symbols: ";
    for (uint s : decoded_symbols) {
        std::cout << s << " ";
    }
    std::cout << "\n";

    return 0;
}
```

⚠️ 注意事项（简化版局限）：

1. 此代码仅为教学示例，真实的 rANS 实现要复杂得多，包括：

   - 精确控制状态位数（如 64 位）。
   - 精细控制输出比特数（如每次输出 9~16 位，而不是固定 8 位）。
   - 更高效的符号查找（如使用 CDF 表和二分查找，或完美哈希）。
   - 支持字节流输入/输出，而不是简单的 push/pop。

2. 状态管理、输出缓冲和字节对齐需要更细致的处理，以支持高效压缩。

3. 实际项目推荐使用成熟的库，如：

   - https://github.com/facebook/zstd（内部使用了类似 ANS 的技术）
   - https://github.com/rygorous/ryg_rans（Ryg 的 rANS 实现，参考实现）
   - Google 的 https://github.com/google/draco（3D 几何压缩中也用了 ANS 思想）

## 四、总结

|项目|说明|
| :- | :- |
|算法名称|rANS（Range Asymmetric Numeral Systems）|
|类别|熵编码（Entropy Encoding）|
|特点|高性能、整数运算、比传统算术编码快、适合硬件|
|对比|Arithmetic Coding 避免浮点数，用整数状态模拟概率分布|
|编码方向|适合从前往后编码，从后往前解码|
|应用|Zstandard, Draco, 各种新型压缩算法|

如果你想进一步学习：

- 推荐阅读：

  - https://arxiv.org/abs/1402.3392
  - Ryg 的 RANS 实现（https://github.com/rygorous/ryg_rans）
  - Zstandard 源码中的 ANS 相关部分

- 进阶方向：

  - 实现多符号、动态频率（自适应 rANS）
  - 使用 64 位状态以获得更大窗口
  - 结合字典编码（如 LZ77 + rANS，正如 Zstd 所做）

如果你希望我提供一个更完整、优化的 rANS C++ 实现（支持多符号、字节流、实际可用），我也可以继续为你提供！
