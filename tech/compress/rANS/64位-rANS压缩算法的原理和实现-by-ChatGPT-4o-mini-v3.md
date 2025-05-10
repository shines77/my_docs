# 64位 rANS 压缩算法的原理和实现

## 问题

请介绍一下压缩算法中的 rANS 范围编码的压缩、解压原理和具体实现，使用 64 位整型版本，使用 C++ 代码示例，并添加中文注释。

## 回答

（来自 ChatGPT-4o-mini，[https://chat.lanjing.pro/](https://chat.lanjing.pro/)）

rANS（Range Asymmetric Numeral System）是一种高效的无损压缩算法。其核心思想是将数字的信息编码与状态传递相结合，通过范围内的动态计算，实现更高效的压缩和解压缩。以下是rANS算法的压缩和解压原理以及C++代码实现示例。

### rANS 压缩原理

1. **状态表示**：rANS使用一个“状态值”来表示当前的压缩状态，通常是一个 64 位的整数。这一状态值会根据编码的输入符号而不断变化。

2. **符号映射**：每个符号都对应一个概率区间，通过这些区间确定符号出现的机会。rANS通过动态调整每个符号的概率范围来实现压缩。

3. **输出结果**：最终的状态值被映射为一定范围内的数字进行输出。

### rANS 解压原理

1. **反向映射**：解压时，从状态值中反向提取符号，利用符号的概率区间来控制解压缩的过程。

2. **更新状态**：解压后，更新状态值以便于处理下一个符号。

### 具体实现示例

以下是使用 C++ 实现的 rANS 算法的简单示例代码，包含中文注释：

```cpp
#include <iostream>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <cassert>

class rANS {
public:
    rANS(const std::unordered_map<char, uint32_t>& probabilities) {
        // 初始化概率
        total_symbols = 0;
        for (auto& pair : probabilities) {
            total_symbols += pair.second;
            symbol_range[pair.first] = total_symbols;
        }
        assert(total_symbols > 0);
    }

    // 压缩方法
    std::vector<uint64_t> compress(const std::string& input) {
        uint64_t state = 1; // 初始化状态
        std::vector<uint64_t> output;

        for (char symbol : input) {
            // 获取符号范围
            uint64_t range = total_symbols - (state % total_symbols);
            uint64_t symbol_prob = symbol_range[symbol];

            // 缩小范围
            state = (state / total_symbols) * symbol_prob;

            // 若状态值过大，进行输出
            while (state >= (1ULL << 32)) {
                output.push_back(state & 0xFFFFFFFF);
                state >>= 32;
            }
        }

        // 在最后输出状态
        output.push_back(state);
        return output;
    }

    // 解压缩方法
    std::string decompress(const std::vector<uint64_t>& input) {
        std::string output;
        uint64_t state = 0;

        // 初始化状态
        for (size_t i = 0; i < input.size(); ++i) {
            state = (state << 32) | input[i]; // 将输入拼接到状态中
            while (true) {
                // 遍历概率区间进行符号解压
                for (auto& pair : symbol_range) {
                    char symbol = pair.first;
                    uint64_t symbol_prob = pair.second;

                    if (state < symbol_prob) {
                        output += symbol;
                        state -= symbol_prob; // 更新状态
                        state *= total_symbols; // 扩展状态
                        break; // 处理下一个符号
                    }
                }
                if (state < total_symbols) {
                    break; // 如果状态小于总范围，结束
                }
            }
        }
        return output;
    }

private:
    uint32_t total_symbols;
    std::unordered_map<char, uint32_t> symbol_range; // 符号概率范围
};

// 测试 rANS 压缩和解压
int main() {
    std::unordered_map<char, uint32_t> probabilities = { {'a', 50}, {'b', 30}, {'c', 20} };
    rANS compressor(probabilities);

    std::string data = "aabacaba"; // 待压缩字符串
    std::vector<uint64_t> compressed = compressor.compress(data); // 执行压缩

    std::cout << "Compressed Data: ";
    for (auto num : compressed) {
        std::cout << num << " ";
    }
    std::cout << std::endl;

    std::string decompressed = compressor.decompress(compressed); // 执行解压
    std::cout << "Decompressed Data: " << decompressed << std::endl;

    return 0;
}
```

### 代码说明

1. **符号概率初始化**：构造函数接受一个字符及其出现概率的哈希表，以初始化符号的概率范围。

2. **压缩逻辑**：在压缩过程中，不断缩小状态值，每当状态值过大时，将其低32位推送到输出。

3. **解压逻辑**：在解压过程中，通过状态值的范围反向映射出符号，形成原始字符串。

### 总结

rANS 是一种高效且灵活的无损压缩算法，能够有效利用符号的概率信息进行压缩和解压。通过合理的编码方式，它在处理复杂数据时展现出了优越的性能。在许多优化场景中，rANS被广泛应用于数据压缩领域。
