# 64位 rANS 压缩算法的原理和实现

## 问题

请介绍一下压缩算法中的 rANS 范围编码的压缩、解压原理和具体实现，使用 64 位整型版本，使用 C++ 代码示例，并添加中文注释。

## 回答

（来自 QWen3-235B-A22B，[https://bailian.console.aliyun.com/](https://bailian.console.aliyun.com/)）

### rANS 范围编码原理与 C++ 实现

#### **1. 原理概述**

rANS（Range Asymmetric Numeral System）是一种基于概率分布的高效熵编码算法，结合了算术编码的压缩效率和霍夫曼编码的速度优势。其核心思想是将符号的概率映射到整数区间上，通过状态变换实现压缩和解压。

**关键概念：**

- **状态（State）**：一个64位整型值，表示当前编码/解码过程中的中间状态。
- **频率表（Frequency Table）**：记录每个符号的出现频率及累积频率，用于状态变换。
- **缩放机制**：当状态超出预设范围时，通过位操作将其缩放回有效区间，并将高位部分写入输出流。

#### **2. 压缩原理**

**步骤：**

1. **初始化状态**：初始状态设为1。

2. **符号编码**：对每个符号，根据其频率和累积频率更新状态：

   ```cpp
   new_state = (state / freq) * total_freq + (cum_freq + state % freq);
   ```

3. **状态缩放**：若新状态超过阈值（如 `2^32`），将低位写入输出流，保留高位继续处理。

4. **最终输出**：编码完成后，将最终状态写入输出流。

#### **3. 解压原理**

**步骤：**

1. **初始化状态**：从输出流末尾读取最终状态作为初始值。

2. **符号解码**：对当前状态，通过模运算确定符号：

   ```cpp
   mod_val = state % total_freq;
   symbol = find_symbol(mod_val); // 根据累积频率表查找符号
   ```

3. **状态恢复**：根据符号的频率和累积频率恢复前一个状态：

   ```cpp
   cum = cum_freq_table[symbol];
   freq = freq_table[symbol];
   state = (state / total_freq) * freq + (mod_val - cum);
   ```

4. **状态补充**：若状态小于阈值（如 `2^32`），从输入流中补充低位。

#### **4. C++ 实现**

以下代码实现了一个完整的 rANS 编码器和解码器，使用 64 位整型，并附有中文注释。

```cpp
#include <iostream>
#include <vector>
#include <cstdint>

// 频率表结构体
struct FrequencyTable {
    std::vector<uint32_t> freqs;      // 每个符号的频率
    std::vector<uint32_t> cum_freqs;  // 累积频率（cum_freqs[i] = sum(freqs[0..i-1])）
    uint32_t total;                   // 总频率（等于所有频率之和）

    // 构造函数：根据频率数组初始化累积频率
    FrequencyTable(const std::vector<uint32_t>& frequencies) {
        freqs = frequencies;
        total = 0;
        cum_freqs.resize(freqs.size());
        for (size_t i = 0; i < freqs.size(); ++i) {
            cum_freqs[i] = total;
            total += freqs[i];
        }
    }

    // 根据模值查找符号
    uint32_t getSymbol(uint32_t mod_val) const {
        for (size_t i = 0; i < cum_freqs.size(); ++i) {
            if (cum_freqs[i] <= mod_val && (i + 1 == cum_freqs.size() || cum_freqs[i + 1] > mod_val)) {
                return static_cast<uint32_t>(i);
            }
        }
        return static_cast<uint32_t>(freqs.size() - 1); // 默认返回最后一个符号
    }
};

// rANS 编码器类
class rANSEncoder {
public:
    rANSEncoder(FrequencyTable& ft) : freq_table(ft) {}

    // 编码一个符号
    void encode(uint32_t symbol) {
        uint32_t cum = freq_table.cum_freqs[symbol];
        uint32_t freq = freq_table.freqs[symbol];

        // 更新状态
        uint64_t q = state / freq;
        uint64_t r = state % freq;
        state = q * freq_table.total + cum + r;

        // 如果状态超过阈值（2^32），将低位写入缓冲区
        while (state >= (1ULL << 32)) {
            buffer.push_back(static_cast<uint8_t>(state & 0xFF));
            buffer.push_back(static_cast<uint8_t>((state >> 8) & 0xFF));
            buffer.push_back(static_cast<uint8_t>((state >> 16) & 0xFF));
            buffer.push_back(static_cast<uint8_t>((state >> 24) & 0xFF));
            state >>= 32;
        }
    }

    // 获取编码后的数据（包括最终状态）
    std::vector<uint8_t> getEncodedData() const {
        std::vector<uint8_t> result = buffer;

        // 将最终状态写入缓冲区（4字节）
        result.push_back(static_cast<uint8_t>(state & 0xFF));
        result.push_back(static_cast<uint8_t>((state >> 8) & 0xFF));
        result.push_back(static_cast<uint8_t>((state >> 16) & 0xFF));
        result.push_back(static_cast<uint8_t>((state >> 24) & 0xFF));

        return result;
    }

private:
    uint64_t state = 1;               // 初始状态
    std::vector<uint8_t> buffer;      // 输出缓冲区
    FrequencyTable& freq_table;       // 频率表
};

// rANS 解码器类
class rANSDecoder {
public:
    rANSDecoder(const std::vector<uint8_t>& data, FrequencyTable& ft)
        : freq_table(ft), buffer(data) {
        // 从数据末尾读取初始状态（4字节）
        size_t pos = buffer.size() - 4;
        state = static_cast<uint64_t>(buffer[pos]) |
                static_cast<uint64_t>(buffer[pos + 1]) << 8 |
                static_cast<uint64_t>(buffer[pos + 2]) << 16 |
                static_cast<uint64_t>(buffer[pos + 3]) << 24;
        buffer.resize(pos); // 移除最后4字节
    }

    // 解码一个符号
    uint32_t decode() {
        // 计算模值
        uint32_t mod_val = static_cast<uint32_t>(state % freq_table.total);

        // 查找符号
        uint32_t symbol = freq_table.getSymbol(mod_val);

        // 恢复前一个状态
        uint32_t cum = freq_table.cum_freqs[symbol];
        uint32_t freq = freq_table.freqs[symbol];
        uint64_t q = state / freq_table.total;
        state = q * freq + (mod_val - cum);

        // 如果状态小于阈值（2^32），从缓冲区补充低位
        while (state < (1ULL << 32) && !buffer.empty()) {
            // 从缓冲区末尾读取4字节
            size_t pos = buffer.size() - 4;
            uint64_t low = 0;
            low |= static_cast<uint64_t>(buffer[pos]);
            low |= static_cast<uint64_t>(buffer[pos + 1]) << 8;
            low |= static_cast<uint64_t>(buffer[pos + 2]) << 16;
            low |= static_cast<uint64_t>(buffer[pos + 3]) << 24;
            buffer.resize(pos); // 移除最后4字节

            state = (state << 32) | low; // 合并低位
        }

        return symbol;
    }

    // 检查是否还有数据
    bool hasMoreData() const {
        return state != 1; // 初始状态为1，解码到初始状态时结束
    }

private:
    uint64_t state;                     // 当前状态
    std::vector<uint8_t> buffer;        // 输入缓冲区
    FrequencyTable& freq_table;         // 频率表
};
```

#### **5. 使用示例**

```cpp
int main() {
    // 定义频率表（假设符号0的频率为3，符号1的频率为5）
    std::vector<uint32_t> frequencies = {3, 5};
    FrequencyTable freq_table(frequencies);

    // 编码测试
    rANSEncoder encoder(freq_table);
    std::vector<uint32_t> input = {0, 1, 0, 1, 1}; // 待编码序列
    for (uint32_t symbol : input) {
        encoder.encode(symbol);
    }
    std::vector<uint8_t> encoded = encoder.getEncodedData();

    // 解码测试
    rANSDecoder decoder(encoded, freq_table);
    std::vector<uint32_t> decoded;
    while (decoder.hasMoreData()) {
        decoded.push_back(decoder.decode());
    }

    // 输出结果
    std::cout << "Decoded sequence: ";
    for (uint32_t symbol : decoded) {
        std::cout << symbol << " ";
    }
    std::cout << std::endl;

    return 0;
}
```

#### **6. 关键点总结**

- **状态管理**：编码时动态调整状态范围，解码时逐步恢复。
- **频率表**：需预先统计符号频率，构建累积频率表。
- **位操作**：使用位移和掩码处理状态缩放，避免溢出。
- **LIFO 特性**：解码顺序与编码顺序相反，需注意数据顺序。

通过上述实现，可以高效地使用 rANS 进行数据压缩和解压，适用于需要高压缩率的场景。
