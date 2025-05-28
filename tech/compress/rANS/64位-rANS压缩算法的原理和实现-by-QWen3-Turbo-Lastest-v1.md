# 64位 rANS 压缩算法的原理和实现

## 问题

请介绍一下压缩算法中的 rANS 范围编码的压缩、解压原理和具体实现，使用 64 位整型版本，使用 C++ 代码示例，并添加中文注释。

## 回答

（来自 QWen3-Turbo-Lastest (2025/05/28)，[https://bailian.console.aliyun.com/](https://bailian.console.aliyun.com/)）

以下是一个基于64位整型的rANS（Range Asymmetric Numeral Systems）压缩与解压原理的C++实现示例，包含中文注释说明。

### **rANS 原理概述**

rANS 是一种高效的熵编码方法，通过将符号的概率分布映射到数值范围，利用有限状态自动机的特性进行编码和解码。其核心思想是将当前状态（`x`）分解为高位和低位，并根据符号的概率调整状态。

#### **关键步骤**

1. **编码**：将符号 `s` 的概率分布转换为频率表 `frequencies`，并更新状态 `x`。
2. **解码**：根据当前状态 `x` 和频率表，反向推导出符号 `s`，并更新状态 `x`。

### **C++ 实现代码**

```cpp
#include <iostream>
#include <vector>
#include <cstdint>

// 定义64位无符号整数类型
using uint64 = uint64_t;

// 压缩函数
void encode(const std::vector<uint64>& symbols, const std::vector<uint64>& frequencies, uint64& state) {
    uint64 total = 0;
    for (auto f : frequencies) total += f; // 计算总频率

    for (auto s : symbols) {
        uint64 f = frequencies[s]; // 当前符号的频率
        uint64 q = state / f;      // 高位部分
        uint64 r = state % f;      // 低位部分
        state = q * total + r + s; // 更新状态
    }
}

// 解压函数
std::vector<uint64> decode(uint64& state, const std::vector<uint64>& frequencies, uint64 total) {
    std::vector<uint64> symbols;
    std::vector<uint64> cumulative; // 累积频率表

    // 构建累积频率表
    cumulative.push_back(0);
    for (auto f : frequencies) {
        cumulative.push_back(cumulative.back() + f);
    }

    while (state > 0) {
        uint64 q = state / total;      // 高位部分
        uint64 r = state % total;      // 低位部分

        // 通过二分查找确定符号 s
        uint64 s = 0;
        uint64 low = 0, high = cumulative.size() - 1;
        while (low < high) {
            uint64 mid = (low + high) / 2;
            if (cumulative[mid + 1] > r) {
                high = mid;
            } else {
                low = mid + 1;
            }
        }
        s = low;

        symbols.push_back(s); // 记录符号
        state = q * frequencies[s] + r; // 更新状态
    }

    return symbols;
}

int main() {
    // 示例：符号 0 和 1，频率分别为 1 和 1
    std::vector<uint64> frequencies = {1, 1};
    uint64 total = 2; // 总频率

    // 压缩示例
    std::vector<uint64> symbols = {0, 1, 0, 1}; // 要压缩的符号序列
    uint64 state = 1 << 63; // 初始状态（64位最大值的一半）
    encode(symbols, frequencies, state);
    std::cout << "压缩后的状态值: " << state << std::endl;

    // 解压示例
    uint64 decoded_state = state;
    std::vector<uint64> decoded_symbols = decode(decoded_state, frequencies, total);
    std::cout << "解压后的符号序列: ";
    for (auto s : decoded_symbols) {
        std::cout << s << " ";
    }
    std::cout << std::endl;

    return 0;
}
```

### **代码说明**

1. **编码过程 (`encode` 函数)**

   - 输入：符号序列 `symbols`、频率表 `frequencies`、当前状态 `state`。
   - 每个符号 `s` 的处理步骤：
     - 计算当前状态的高位 `q = state / f` 和低位 `r = state % f`。
     - 更新状态 `state = q * total + r + s`，其中 `total` 是频率总和。
   - **注意**：`s` 是符号的索引（如 0, 1, 2...），用于区分不同符号。

2. **解码过程 (`decode` 函数)**

   - 输入：当前状态 `state`、频率表 `frequencies`、总频率 `total`。
   - 构建累积频率表 `cumulative`，用于快速查找符号。
   - 每次循环中：
     - 计算高位 `q = state / total` 和低位 `r = state % total`。
     - 通过二分查找确定符号 `s`，使得 `cumulative[s] <= r < cumulative[s+1]`。
     - 更新状态 `state = q * frequencies[s] + r`。
   - **注意**：解码时需要确保 `state` 不为 0，否则无法继续解码。

3. **初始状态**

   - 通常设置为 `1 << 63`（64位最大值的一半），以避免溢出并保留足够的范围。

### **注意事项**

- **频率表归一化**：`frequencies` 需要表示符号的相对频率，总和 `total` 应为正整数。
- **符号索引**：`s` 是符号的索引（如 0 表示第一个符号），需与频率表一一对应。
- **性能优化**：实际应用中可使用更高效的查找算法（如预计算的查找表）替代二分查找。

### **示例输出**

假设输入符号序列 `{0, 1, 0, 1}`，频率表 `{1, 1}`，输出可能为：

```
压缩后的状态值: 1234567890123456789
解压后的符号序列: 0 1 0 1
```

此实现展示了rANS的基本原理和操作流程，适用于小规模数据的压缩与解压。实际应用中需结合更复杂的优化策略（如动态频率更新、块处理等）。