# 64位 rANS 压缩算法的原理和实现

## 问题

请介绍一下压缩算法中的 rANS 范围编码的压缩、解压原理和具体实现，使用 64 位整型版本，使用 C++ 代码示例，并添加中文注释。

## 回答

（来自 QWen3-Plus-Lastest (2025/05/28)，[https://bailian.console.aliyun.com/](https://bailian.console.aliyun.com/)）

### rANS 范围编码原理与 C++ 实现

rANS（Range Asymmetric Numeral System）是一种高效的熵编码方法，结合了算术编码的压缩率和类似霍夫曼编码的快速实现。其核心思想是将符号的概率分布映射到整数状态中，并通过位操作维护状态范围以避免溢出。

#### **压缩原理**

1. **状态表示**：使用 64 位整型 `x` 表示当前状态，初始值为 1。

2. **符号编码**：

   - 对每个符号 `s`，根据其频率 `f_s` 和累积频率 `C_s`，将状态 `x` 映射为新状态：

     ```cpp
     x = (x / f_s) * F + (x % f_s) + C_s;
     ```
     其中 `F` 是总频率，`C_s` 是符号 `s` 的累积频率。

3. **归一化**：在编码前确保 `x` 在允许范围内（如 `[L, 2L)`），否则将低位写入输出流并缩小 `x`。

#### **解压原理**

1. **状态恢复**：从输入流读取位填充 `x`，直到 `x >= L`。

2. **符号解码**：

   - 计算 `bucket = x % F`，找到对应的符号 `s`。
   - 更新状态 `x` 为前一个状态：

     ```cpp
     x = (x / F) * f_s + (bucket - C_s);
     ```

3. **归一化**：在解码后若 `x < L`，从输入流读取位补充 `x`。

#### **C++ 实现**

```cpp
#include <iostream>
#include <vector>
#include <cstdint>

// 假设符号集为 {A, B, C}，频率分别为 3, 3, 2，总频率 F = 8
const int F = 8;  // 总频率（需为 2 的幂以简化实现）
const uint64_t L = 1ULL << 32;  // 状态下限

// 累积频率表（C_s）和频率表（f_s）
const std::vector<int> freq = {3, 3, 2};  // 符号 A/B/C 的频率
const std::vector<int> C = {0, 3, 6};     // 累积频率

// 编码单个符号
void encode(uint64_t &x, int symbol, std::vector<uint8_t> &out) {
    int f_s = freq[symbol];
    int C_s = C[symbol];

    // 归一化：若 x >= (L / F) * f_s，则输出低位并缩小 x
    while (x >= (L / F) * f_s) {
        out.push_back(static_cast<uint8_t>(x));  // 写入低位字节
        x >>= 8;  // 右移 8 位（模拟输出 8 bits）
    }

    // 编码公式：x' = (x/f_s)*F + (x%f_s) + C_s
    x = (x / f_s) * F + (x % f_s) + C_s;
}

// 解码单个符号
int decode(uint64_t &x, std::vector<uint8_t>::iterator &it, std::vector<uint8_t>::iterator end) {
    // 当 x < L 时，从输入流读取字节填充高位
    while (x < L && it != end) {
        x = (x << 8) | (*it++);  // 左移 8 位并填充新字节
    }

    // 计算 bucket = x % F
    int bucket = x % F;

    // 查找符号（简单线性搜索）
    int symbol = -1;
    for (size_t i = 0; i < C.size(); ++i) {
        if (C[i] <= bucket && (i + 1 == C.size() || bucket < C[i + 1])) {
            symbol = i;
            break;
        }
    }

    // 更新状态：x_prev = (x/F)*f_s + (bucket - C_s)
    int f_s = freq[symbol];
    int C_s = C[symbol];
    x = (x / F) * f_s + (bucket - C_s);

    return symbol;
}

int main() {
    // 示例输入数据（符号序列：A=0, B=1, C=2）
    std::vector<int> input = {0, 1, 2, 1, 0};

    // 编码过程
    uint64_t x = 1;  // 初始状态
    std::vector<uint8_t> compressed;
    for (int s : input) {
        encode(x, s, compressed);
    }
    // 最终输出剩余状态
    while (x > 0) {
        compressed.push_back(static_cast<uint8_t>(x));
        x >>= 8;
    }

    std::cout << "压缩后的字节流：";
    for (uint8_t b : compressed) std::cout << std::hex << +b << " ";
    std::cout << std::endl;

    // 解码过程
    x = 1;  // 重置状态
    std::vector<int> decoded;
    auto it = compressed.begin();
    auto end = compressed.end();
    while (it != end) {
        int s = decode(x, it, end);
        if (s != -1) decoded.push_back(s);
    }

    std::cout << "解压后的符号序列：";
    for (int s : decoded) std::cout << s << " ";
    std::cout << std::endl;

    return 0;
}
```

#### **代码说明**

1. **频率与累积分布**：

   - `freq`：符号频率数组。
   - `C`：累积频率数组，用于确定符号在 `bucket` 中的范围。

2. **归一化操作**：

   - **编码**：当 `x` 过大时，输出低位字节并缩小 `x`。
   - **解码**：当 `x` 过小时，从输入流读取字节补充高位。

3. **位操作**：

   - 使用 `>> 8` 和 `<< 8` 模拟字节级输入/输出。

4. **复杂度优化**：

   - 查找符号使用线性搜索，实际应用中可用二分查找或预计算表优化。

#### **注意事项**

- **频率总和**：`F` 需为 2 的幂以简化归一化计算。
- **状态范围**：`L` 通常设为 `2^32` 以充分利用 64 位整型。
- **性能优化**：实际实现中需使用更高效的符号查找和流缓冲策略。
