# 64位 rANS 压缩算法的原理和实现

## 问题

请介绍一下压缩算法中的 rANS 范围编码的压缩、解压原理和具体实现，使用 64 位整型版本，使用 C++ 代码示例，并添加中文注释。

## 回答

（来自 Grok-3，[https://chat.lanjing.pro/](https://chat.lanjing.pro/)）

### 简介

rANS（Range Asymmetric Numeral Systems，范围非对称数字系统）是一种高效的熵编码算法，广泛应用于现代压缩技术中，如zstd和JPEG XL。rANS的核心思想是将输入符号的概率分布映射到一个整数范围，通过迭代更新编码状态来实现压缩。以下是对rANS的压缩与解压原理的详细介绍，并附带使用64位整型的C++代码实现，包含中文注释。

### rANS压缩与解压原理

#### 1. **基本概念**

- **符号与概率**：rANS假设输入是一系列符号（例如字节），每个符号有其出现概率。概率分布通常通过预计算的累积分布函数（CDF）表示。
- **编码状态**：rANS使用一个64位整数`state`表示编码状态。压缩时，`state`被更新以嵌入输入符号的信息；解压时，从`state`中提取符号并恢复原始数据。
- **范围编码**：rANS将符号的概率映射到整数范围，范围的大小由`freq`（频率）和`scale_bits`（精度位数）决定。编码过程中，`state`被压缩到一个较小的范围，解码时则反向操作。

#### 2. **压缩过程**

压缩是将输入符号编码到`state`中：

1. **初始化**：`state`从一个初始值开始（通常是大整数）。
2. **符号编码**：
   - 对于每个输入符号，查找其概率分布（通过CDF和频率表）。
   - 更新`state`：`state = (state / freq) * (1 << scale_bits) + (state % freq) + cdf`，其中`freq`是符号的频率，`cdf`是累积分布，`scale_bits`是范围的精度。
3. **归一化**：当`state`超过某个阈值时，将低位输出到压缩流中，并将`state`右移，以保持在有效范围内。
4. **结束**：将最终的`state`写入输出流。

#### 3. **解压过程**

解压是从`state`中恢复原始符号：

1. **初始化**：从压缩流中读取初始`state`。
2. **符号解码**：
   - 提取`state`的低位（`slot = state & ((1 << scale_bits) - 1)`），通过查找表找到对应的符号、频率和累积分布。
   - 更新`state`：`state = freq * (state >> scale_bits) + (slot - cdf)`。
3. **归一化**：当`state`低于某个阈值时，从输入流中读取更多位，左移补充到`state`中。
4. **结束**：重复直到解码所有符号。

#### 4. **关键点**

- **概率表**：rANS需要预先构建符号的概率分布表，包括频率（`freq`）和累积分布（`cdf`）。
- **归一化**：压缩和解压都需要处理`state`的溢出或不足，通过输出/输入字节流来保持`state`在有效范围内。
- **性能优化**：rANS的性能依赖于快速的查找表和位操作，使用64位整数可以提供足够的范围，减少归一化操作的频率。

### C++代码实现

以下是一个完整的C++实现，包含rANS的压缩和解压功能，使用64位整型，假设输入是字节（0-255），并使用简单的静态概率模型。

```cpp
#include <cstdint>
#include <vector>
#include <iostream>

// rANS 编码器和解码器
class rANS {
private:
    static constexpr int kScaleBits = 12; // 范围精度，2^12 = 4096
    static constexpr uint64_t kLowerBound = 1ULL << 31; // 归一化下界
    static constexpr uint64_t kMask = (1ULL << kScaleBits) - 1; // 范围掩码

    // 概率表结构
    struct SymbolInfo {
        uint32_t freq; // 符号频率
        uint32_t cdf;  // 累积分布
    };

    std::vector<SymbolInfo> encode_table; // 编码表
    std::vector<uint8_t> decode_table;    // 解码表（slot -> 符号）

public:
    rANS() {
        // 初始化概率表（示例：均匀分布）
        encode_table.resize(256);
        decode_table.resize(1 << kScaleBits);
        uint32_t total_freq = 1 << kScaleBits;
        for (int i = 0; i < 256; ++i) {
            encode_table[i].freq = total_freq / 256; // 每个符号频率
            encode_table[i].cdf = i * (total_freq / 256); // 累积分布
        }
        // 构建解码表
        for (uint32_t slot = 0; slot < (1 << kScaleBits); ++slot) {
            for (int i = 0; i < 256; ++i) {
                if (encode_table[i].cdf <= slot && slot < encode_table[i].cdf + encode_table[i].freq) {
                    decode_table[slot] = static_cast<uint8_t>(i);
                    break;
                }
            }
        }
    }

    // 压缩函数
    std::vector<uint8_t> compress(const std::vector<uint8_t>& input) {
        std::vector<uint8_t> output;
        uint64_t state = kLowerBound;

        // 从后向前编码（便于解码时从前向后读取）
        for (size_t i = input.size(); i > 0; --i) {
            uint8_t symbol = input[i - 1];
            uint32_t freq = encode_table[symbol].freq;
            uint32_t cdf = encode_table[symbol].cdf;

            // rANS 编码公式
            state = (state / freq) * (1ULL << kScaleBits) + (state % freq) + cdf;

            // 归一化：当 state 过大时，输出低位字节
            while (state >= kLowerBound) {
                output.push_back(static_cast<uint8_t>(state & 0xFF));
                state >>= 8;
            }
        }

        // 输出最终状态
        while (state > 0) {
            output.push_back(static_cast<uint8_t>(state & 0xFF));
            state >>= 8;
        }

        // 反转输出（因为编码是从后向前）
        std::reverse(output.begin(), output.end());
        return output;
    }

    // 解压函数
    std::vector<uint8_t> decompress(const std::vector<uint8_t>& input, size_t output_size) {
        std::vector<uint8_t> output(output_size);
        uint64_t state = 0;

        // 初始化状态
        size_t input_pos = 0;
        for (int i = 0; i < 8 && input_pos < input.size(); ++i) {
            state = (state << 8) | input[input_pos++];
        }

        // 解码
        for (size_t i = 0; i < output_size; ++i) {
            // 提取 slot
            uint32_t slot = state & kMask;
            uint8_t symbol = decode_table[slot];
            uint32_t freq = encode_table[symbol].freq;
            uint32_t cdf = encode_table[symbol].cdf;

            // 更新状态
            state = freq * (state >> kScaleBits) + (slot - cdf);

            // 归一化：当 state 过小时，读取更多字节
            while (state < kLowerBound && input_pos < input.size()) {
                state = (state << 8) | input[input_pos++];
            }

            output[i] = symbol;
        }

        return output;
    }
};

// 测试代码
int main() {
    rANS rans;
    std::vector<uint8_t> input = {'H', 'e', 'l', 'l', 'o'};

    // 压缩
    auto compressed = rans.compress(input);
    std::cout << "压缩数据大小: " << compressed.size() << " 字节
";

    // 解压
    auto decompressed = rans.decompress(compressed, input.size());
    std::cout << "解压结果: ";
    for (uint8_t c : decompressed) {
        std::cout << c;
    }
    std::cout << "\n";

    return 0;
}
```

### 代码说明

1. **概率表**：

   - `encode_table`存储每个符号的频率和累积分布，假设均匀分布（每个字节频率为`4096/256`）。
   - `decode_table`将`slot`映射到符号，用于快速解码。

2. **压缩**：

   - 从后向前编码输入符号，更新`state`并在溢出时输出字节。
   - 最终`state`被写入输出流。

3. **解压**：

   - 从输入流读取初始`state`，通过查找`decode_table`恢复符号。
   - 更新`state`并在不足时读取更多字节。

4. **参数选择**：

   - `kScaleBits = 12`表示范围精度为4096，适合大多数应用。
   - `kLowerBound = 1ULL << 31`确保归一化操作的效率。

### 优化与扩展

1. **动态概率模型**：实际应用中，概率表可以通过自适应模型（如上下文建模）动态更新，以提高压缩率。
2. **性能优化**：可以使用SIMD指令或预计算表来加速编码和解码。
3. **错误处理**：生产环境中需要添加输入验证和错误处理机制。

### 总结

rANS通过高效的范围编码和归一化操作，结合概率分布，实现了高性能的压缩和解压。上述C++实现展示了64位整型版本的核心逻辑，适合学习和实验。实际应用中，可根据具体需求调整概率模型和优化实现。
