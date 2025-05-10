# 64位 rANS 压缩算法的原理和实现

## 问题

请介绍一下压缩算法中的 rANS 范围编码的压缩、解压原理和具体实现，使用 64 位整型版本，使用 C++ 代码示例，并添加中文注释。

## 回答

（来自 ChatGPT-4o-mini，[https://chat.lanjing.pro/](https://chat.lanjing.pro/)）

### rANS 范围编码原理

rANS（Range Adaptive Non-Uniform Symbol）是一种基于范围编码的高效无损压缩算法。它结合了算术编码的优点，通过动态调整符号的概率分布来实现压缩。rANS特有的模式使得它在处理短符号序列时能够获得更高的压缩率。

#### 原理

1. **概率模型**：rANS 使用上下文相关的概率模型来动态调整每个符号的概率。每个符号的概率会在编码过程中不断更新。

2. **范围编码**：与传统的算术编码类似，rANS 通过维护一个当前范围来表示信息。随着符号的逐步被编码或解码，范围会被细化。

3. **整型表示**：rANS 使用整型数值（如64位整型）来表示当前状态，这种方法在实现时更高效，避免了浮点数运算。

4. **输出**：最终结果在输出时将范围的低位部分作为压缩数据，解压时则使用相反的过程。

### C++ 代码实现

以下是 rANS 的一个简单实现，使用 64 位整型进行编码和解码。

```cpp
#include <iostream>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <cassert>

class rANS {
public:
    rANS() : state(0), output(0), totalSymbols(0) {}

    // 添加符号及其频率
    void setSymbolFrequency(const std::unordered_map<char, int>& frequencies) {
        this->frequencies = frequencies;
        totalSymbols = 0;
        for (const auto& pair : frequencies) {
            totalSymbols += pair.second;
        }
        buildCumulativeFrequency();
    }

    // 编码数据
    void encode(const std::string& data) {
        for (char symbol : data) {
            assert(frequencies.find(symbol) != frequencies.end()); // 确保符号在频率表中
            // 获取符号的累积频率
            uint64_t cumFreq = cumulativeFrequency[symbol];
            uint64_t freq = frequencies[symbol];
            // 计算新的范围
            uint64_t range = state / totalSymbols; // 获取当前范围
            state = (state % totalSymbols) + (cumFreq * range) / totalSymbols; // 更新状态
            // 输出当前状态的低位
            while (state >= (1ULL << 32)) {
                outputByte((uint32_t)(state & 0xFFFFFFFF)); // 输出低32位
                state >>= 32; // 左移状态，使其变小
            }
        }
    }

    // 解码数据
    std::string decode(size_t length) {
        std::string result;
        for (size_t i = 0; i < length; ++i) {
            // 计算当前范围
            uint64_t range = state / totalSymbols;
            uint64_t scaledValue = (state % totalSymbols) * (range / totalSymbols);
            // 根据累计频率找到当前符号
            char symbol = findSymbol(scaledValue);
            result += symbol;
            // 更新状态
            uint64_t cumFreq = cumulativeFrequency[symbol];
            uint64_t freq = frequencies[symbol];
            state = (state % totalSymbols) + (cumFreq * range) / totalSymbols; // 更新状态
            // 再次处理输出
            while (state >= (1ULL << 32)) {
                state >>= 32;
            }
        }
        return result;
    }

    // 输出字节（模拟输出操作）
    void outputByte(uint32_t byte) {
        // 在实际中，您可能会将字节写入文件或其他输出流
        std::cout << "Output byte: " << byte << std::endl;
    }

private:
    uint64_t state; // 当前状态
    uint64_t output; // 输出结果（未实际使用，此处可用于扩展功能）
    int totalSymbols; // 总符号数
    std::unordered_map<char, int> frequencies; // 符号频率映射
    std::unordered_map<char, uint64_t> cumulativeFrequency; // 累计频率图

    // 构建累计频率表
    void buildCumulativeFrequency() {
        cumulativeFrequency.clear();
        uint64_t cumFreq = 0;
        for (const auto& pair : frequencies) {
            cumulativeFrequency[pair.first] = cumFreq;
            cumFreq += pair.second;
        }
    }

    // 根据标量值找到符号
    char findSymbol(uint64_t value) {
        for (const auto& pair : frequencies) {
            if (value < cumulativeFrequency[pair.first] + pair.second) {
                return pair.first;
            }
        }
        return '\0'; // 找不到返回空字符
    }
};

int main() {
    rANS encoder;

    // 定义符号及其频率
    std::unordered_map<char, int> frequencies = {
        {'a', 5}, {'b', 3}, {'c', 2}
    };
    encoder.setSymbolFrequency(frequencies);

    // 编码示例数据
    std::string data = "aabcbc";
    encoder.encode(data);

    // 解码数据（示例中为手动设置解码长度）
    std::string decodedData = encoder.decode(data.length());
    std::cout << "Decoded Data: " << decodedData << std::endl;

    return 0;
}
```

### 代码说明

1. **类定义**：

   - `rANS` 类定义了 rANS 编码和解码的功能，包含状态管理和频率表管理。

2. **setSymbolFrequency 方法**：

   - 接受符号频率的映射，计算总符号数，并构建累计频率表。

3. **encode 方法**：

   - 逐个处理输入数据符号，更新状态，并在状态过大时输出其低32位。

4. **decode 方法**：

   - 根据当前状态解码指定长度的数据，通过累计频率找到对应的符号。

5. **private 方法**：

   - `buildCumulativeFrequency` 用于建立累计频率表，有助于快速查找符号。
   - `findSymbol` 方法根据给定的数值找到对应的符号。

6. **main 函数**：

   - 定义频率并进行编码解码的示例，验证编码解码的有效性。

### 总结

rANS 是一种高效的无损压缩算法，通过整型表示管理状态，使得编码和解码过程紧凑且快速。上述实现提供了 rANS 的基础框架，实际应用中可以结合文件输入输出和更复杂的频率模型来扩展其功能。
