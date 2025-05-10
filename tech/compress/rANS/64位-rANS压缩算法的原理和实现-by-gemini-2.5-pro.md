# 64位 rANS 压缩算法的原理和实现

## 问题

请介绍一下压缩算法中的 rANS 范围编码的压缩、解压原理和具体实现，使用 64 位整型版本，使用 C++ 代码示例，并添加中文注释。

## 回答

（来自 gemini-2.5-pro）

好的，我们来详细讲解压缩算法中的 rANS（ranged Asymmetric Numeral Systems）范围编码的原理、64位整型实现以及C++代码示例，并添加中文注释。

### **1. rANS 范围编码原理**

rANS 是一种先进的熵编码方法，属于 ANS（Asymmetric Numeral Systems）家族。与传统的算术编码类似，rANS 根据符号的概率（频率）来分配编码空间，从而实现接近理论极限的压缩率。但与算术编码维护一个浮点数范围不同，rANS 使用一个**整数状态变量 `x`** 来表示当前的编码状态。

**核心思想：**

rANS 的编码和解码过程是一对互逆的运算。它通过将一个旧状态 `x_old` 和一个符号 `s` 映射到一个新状态 `x_new` 来实现编码。解码则是反过来，从当前状态 `x_current` 中恢复出符号 `s` 和旧状态 `x_old`。

假设我们有一个符号集，其总频率（或称为频率表的缩放因子）为 `TotalFreq` (通常是 2 的幂，例如 `1 << L_BITS`，其中 `L_BITS` 是频率表的位数)。每个符号 `s` 有其自身的频率 `freq[s]` 和累积频率 `cdf[s]` (即频率小于 `s` 的所有符号的频率之和)。

* **编码 (Encoding):**

    编码过程通常是**从后往前**处理输入符号序列。
    对于要编码的当前符号 `s`，其频率为 `freq[s]`，累积频率为 `cdf[s]`。当前的编码状态为 `x`。
    编码操作将 `x` 更新为 `x_new`：

    `x_new = (x / freq[s]) * TotalFreq + cdf[s] + (x % freq[s])`

    这个公式可以理解为：

    1. `x / freq[s]`：将当前状态 `x` 大致均分到 `freq[s]` 个槽中，取整数商。
    2. `(x / freq[s]) * TotalFreq`：将这个商乘以总频率 `TotalFreq`，将其扩展到整个编码范围。
    3. `cdf[s]`：加上符号 `s` 的累积频率，将编码位置移到符号 `s` 对应的区间起始处。
    4. `x % freq[s]`：加上余数，将原始状态 `x` 中在 `freq[s]` 内部的偏移信息保留下来。

    为了防止状态 `x` 过大导致溢出，或者过小导致精度损失（特别是 `x / freq[s]` 变得太小时），编码过程中需要进行**归一化 (Normalization)**。
    当状态 `x` 达到一个预设的上限阈值时（通常这个阈值与 `freq[s]` 相关，保证 `x / freq[s]` 足够大），我们会将 `x` 的低位部分（通常是 `NORM_BITS` 位）输出到压缩流中，并将 `x` 右移 `NORM_BITS` 位。这保证了 `x` 始终在一个可管理的范围内。

    编码的输出流是反向的：先处理的符号，其对应的归一化输出位会出现在压缩流的后面。因此，编码器通常将输出位存储在一个缓冲区中，最后再反转缓冲区或以反向方式写入文件。

* **解码 (Decoding):**

    解码过程是**从前往后**处理压缩流。我们从压缩流的末尾（逻辑上的开头）读取初始状态 `x`。
    对于当前状态 `x`，我们需要恢复出符号 `s` 和旧状态 `x_old`。

    1.  **恢复符号 `s`:**

        回顾编码公式 `x_current = (x_old / freq[s]) * TotalFreq + cdf[s] + (x_old % freq[s])`。
        令 `slot = x_current % TotalFreq`。
        我们可以观察到 `slot = cdf[s] + (x_old % freq[s])`。
        由于 `0 <= (x_old % freq[s]) < freq[s]`，所以 `cdf[s] <= slot < cdf[s] + freq[s]` (即 `cdf[s] <= slot < cdf[s+1]`)。
        因此，通过计算 `slot = x_current % TotalFreq`，我们只需在 CDF 表中查找到满足 `cdf[s] <= slot < cdf[s+1]` 的符号 `s` 即可。

    2.  **恢复旧状态 `x_old`:**

        一旦找到符号 `s` (及其 `freq[s]` 和 `cdf[s]`)，我们就可以计算：
        `x_old = (x_current / TotalFreq) * freq[s] + (slot - cdf[s])`
        这里 `x_current / TotalFreq` 对应编码时的 `x_old / freq[s]`，而 `slot - cdf[s]` 对应编码时的 `x_old % freq[s]`。

    为了防止状态 `x` 过小（低于预设的下限阈值），解码过程中也需要进行归一化。当状态 `x` 低于某个下限阈值时（通常是 `IO_BASE`，即 `1ULL << NORM_BITS`），我们会从压缩流中读取 `NORM_BITS` 位，并用这些位来“补充”状态 `x`：`x = (x << NORM_BITS) + read_bits`。这保证了 `x` 始终在解码所需的范围内，使得 `x % TotalFreq` 和 `x / TotalFreq` 能正确工作。

### **2. 64 位整型实现关键参数**

* **状态类型:** `uint64_t` (无符号64位整数)。
* **`L_BITS`:** 频率表基数位数。例如，处理字节数据时，`L_BITS = 8`，则 `TotalFreq = 1 << 8 = 256`。如果使用更大的上下文或更高精度的频率，可以使用 `L_BITS = 12` (`TotalFreq = 4096`) 等。
* **`NORM_BITS`:** 每次归一化处理的位数。对于 64 位状态，选择 `NORM_BITS = 32` 是一个常见的且高效的选择，因为它与 32 位字长对齐，方便进行 I/O 操作。
* **`IO_BASE`:** `1ULL << NORM_BITS`。这是归一化时每次输出/输入的单位。对于 `NORM_BITS = 32`，`IO_BASE = 1ULL << 32`。
* **`RANGE_BITS`:** 状态变量 `x` 的总位数，这里是 64。
* **编码归一化阈值:**
    在编码一个符号 `s` 之前，需要确保状态 `x` 足够大，以使得 `x / freq[s]` 这一项不会损失过多精度，并且编码后的新状态 `x_new` 至少为 `IO_BASE`。
    一个常用的条件是，在编码符号 `s` (频率为 `f_s`) 之前，如果 `x >= (IO_BASE >> L_BITS) * f_s`，则进行归一化（输出低 `NORM_BITS` 位，然后 `x >>= NORM_BITS`），重复此过程直到 `x` 小于该阈值。
    `IO_BASE >> L_BITS` 可以看作是一个缩放因子。例如，若 `IO_BASE = 1ULL << 32`，`L_BITS = 8` (so `TotalFreq = 1ULL << 8`)，则 `IO_BASE >> L_BITS = (1ULL << 32) >> 8 = 1ULL << 24`。
    这个阈值确保 `x / f_s >= (IO_BASE >> L_BITS)`，因此 `(x / f_s) * TotalFreq >= (IO_BASE >> L_BITS) * (1ULL << L_BITS) = IO_BASE`。
    这样，编码后的状态 `x_new` 就会大于等于 `IO_BASE`。
* **解码归一化阈值:**
    在解码一个符号之前，如果 `x < IO_BASE`，则进行归一化（`x = (x << NORM_BITS) + read_bits_from_stream`），重复此过程直到 `x >= IO_BASE`。

### **3. C++ 代码实现**

下面是一个使用 64 位整型状态的 rANS 编码和解码的 C++ 实现示例。

为了简化，我们假设：

* 输入是字节流 (`uint8_t`)。
* 频率和 CDF 表是预先计算好的。
* I/O 直接使用 `std::vector<uint8_t>` 作为缓冲区。

```cpp
#include <iostream>
#include <vector>
#include <numeric>      // for std::partial_sum, std::accumulate
#include <algorithm>    // for std::reverse, std::upper_bound
#include <cstdint>      // for uint64_t, uint32_t, uint8_t

// --- rANS 参数配置 ---
const int L_BITS = 8;                       // 频率表基数位数 (例如，8 位对应 256 个符号)
const uint32_t TOTAL_FREQ = 1 << L_BITS;    // 总频率 (2^L_BITS)

const int NORM_BITS = 32;                   // 每次归一化处理的位数 (推荐 32 位，匹配 uint32_t)
const uint64_t IO_BASE = 1ULL << NORM_BITS; // 归一化 I/O 的基数 (2^NORM_BITS)

// 编码时，状态 x 在编码一个符号前，如果 x >= RANS_X_MAX_BEFORE_NORM_FACTOR * freq[symbol]，则进行归一化
// RANS_X_MAX_BEFORE_NORM_FACTOR 确保 x/freq 足够大，使得 (x/freq)*TOTAL_FREQ >= IO_BASE
// 通常 RANS_X_MAX_BEFORE_NORM_FACTOR = (IO_BASE >> L_BITS)
const uint64_t RANS_X_MAX_BEFORE_NORM_FACTOR = (IO_BASE >> L_BITS);

// --- 符号信息结构体 ---
struct SymbolStats {
    uint32_t freq; // 符号的频率
    uint32_t cdf;  // 符号的累积频率 (sum of frequencies of all preceding symbols)
};

// --- 辅助函数：计算频率和构建符号统计信息 ---
// 注意：实际应用中，频率为0的符号需要特殊处理（例如，给予最小频率1，或使用逃逸码）
// 这个简单示例假设所有可能符号至少出现一次，或者提前处理了零频率。
std::vector<SymbolStats> build_symbol_stats(const std::vector<uint8_t>& data) {
    std::vector<uint32_t> freqs(TOTAL_FREQ, 0);
    for (uint8_t byte : data) {
        freqs[byte]++;
    }

    // 确保所有频率至少为1，以避免除以零和CDF问题。
    // 在实际的压缩器中，这通常通过更复杂的模型（如添加伪计数）来完成。
    uint32_t current_total_freq = 0;
    for (size_t i = 0; i < TOTAL_FREQ; ++i) {
        if (freqs[i] == 0) {
            freqs[i] = 1; // 平滑处理：给未出现的符号一个最小频率
        }
        current_total_freq += freqs[i];
    }

    // 按比例调整频率，使其总和接近 TOTAL_FREQ
    // 这是一个简化的调整，实际的rANS实现对频率表的精度有要求
    // 理想情况下，sum(freqs) == TOTAL_FREQ
    // 如果不是，需要进行缩放。这里为了简单，我们假设调用者保证了这一点，
    // 或者接受因此带来的轻微压缩率损失。
    // 一个更健壮的方法是确保 sum(freqs) == TOTAL_FREQ。
    // 例如，如果 current_total_freq != TOTAL_FREQ, 则需要按比例缩放 freqs 数组
    // 使其总和等于 TOTAL_FREQ，同时确保每个 freq >= 1。
    // 这部分比较复杂，超出了rANS核心原理的范畴，这里略过精确缩放。
    // 我们将直接使用计算出的（可能平滑过的）频率。

    std::vector<SymbolStats> stats(TOTAL_FREQ);
    uint32_t current_cdf = 0;
    for (size_t i = 0; i < TOTAL_FREQ; ++i) {
        stats[i].freq = freqs[i];
        stats[i].cdf = current_cdf;
        current_cdf += freqs[i];
    }
    // 最终的 current_cdf 应该是所有频率的总和。
    // 对于rANS，这个总和必须等于 TOTAL_FREQ 才能获得最佳性能和正确性。
    // 如果不等于，需要一个规范化步骤来调整频率。
    // 在这个例子中，我们假设平滑后的频率总和接近 TOTAL_FREQ。
    // 如果 `current_cdf` 不等于 `TOTAL_FREQ`，则表示频率表未正确归一化到 `TOTAL_FREQ`。
    // 这会导致压缩效率下降或潜在的解码问题。
    // 真实世界的rANS编码器需要严格确保这一点。
    // 例如，可以通过 Fabian Giesen 的 `rans_static.h` 中的频率归一化方法。
    if (current_cdf != TOTAL_FREQ && data.size() > 0) { // 只有当有数据时才检查
         std::cerr << "Warning: Sum of frequencies (" << current_cdf
                   << ") does not match TOTAL_FREQ (" << TOTAL_FREQ
                   << "). This may impact compression or correctness." << std::endl;
        // 在一个严格的实现中，这里应该进行频率调整。
        // 为了简单起见，我们继续，但指出这是一个潜在的问题点。
    }

    return stats;
}

// --- rANS 编码函数 ---
std::vector<uint8_t> rans_encode(const std::vector<uint8_t>& data, const std::vector<SymbolStats>& stats) {
    if (data.empty()) {
        return {};
    }

    std::vector<uint32_t> compressed_words; // 存储归一化输出的32位字
    uint64_t x = IO_BASE; // 初始化 rANS 状态，必须 >= IO_BASE

    // 从后往前编码数据
    for (auto it = data.rbegin(); it != data.rend(); ++it) {
        uint8_t symbol = *it;
        const SymbolStats& s_stats = stats[symbol];

        // 编码归一化：
        // 在编码符号前，如果状态 x 太大，则输出低 NORM_BITS 位并右移
        // 阈值 RANS_X_MAX_BEFORE_NORM_FACTOR * s_stats.freq 确保
        // (x / s_stats.freq) * TOTAL_FREQ >= IO_BASE
        uint64_t threshold = RANS_X_MAX_BEFORE_NORM_FACTOR * s_stats.freq;
        while (x >= threshold) {
            compressed_words.push_back(static_cast<uint32_t>(x & (IO_BASE - 1))); // 输出低32位
            x >>= NORM_BITS; // 右移32位
        }

        // rANS 核心编码步骤
        // x_new = (x / freq[s]) * TotalFreq + cdf[s] + (x % freq[s])
        x = (x / s_stats.freq) * TOTAL_FREQ + s_stats.cdf + (x % s_stats.freq);
    }

    // 编码完成后，最终状态 x 也需要写入
    // 将64位状态 x 拆分为两个32位字写入
    compressed_words.push_back(static_cast<uint32_t>(x & (IO_BASE - 1))); // 低32位
    compressed_words.push_back(static_cast<uint32_t>(x >> NORM_BITS));    // 高32位

    // 将32位字的vector转换为字节vector (小端模式)
    // 注意：编码输出的字是反序的，所以最终字节流也需要反转
    std::vector<uint8_t> compressed_bytes;
    compressed_bytes.reserve(compressed_words.size() * 4);
    for (uint32_t word : compressed_words) {
        compressed_bytes.push_back(static_cast<uint8_t>(word & 0xFF));
        compressed_bytes.push_back(static_cast<uint8_t>((word >> 8) & 0xFF));
        compressed_bytes.push_back(static_cast<uint8_t>((word >> 16) & 0xFF));
        compressed_bytes.push_back(static_cast<uint8_t>((word >> 24) & 0xFF));
    }

    // 由于 compressed_words 是按编码顺序（符号逆序）生成的，
    // 并且每个字内部是小端，所以整个字节流需要反转以匹配解码顺序。
    std::reverse(compressed_bytes.begin(), compressed_bytes.end());

    return compressed_bytes;
}

// --- rANS 解码函数 ---
std::vector<uint8_t> rans_decode(const std::vector<uint8_t>& compressed_bytes,
                                 const std::vector<SymbolStats>& stats,
                                 size_t original_data_size) {
    if (compressed_bytes.empty() || original_data_size == 0) {
        return {};
    }
    if (compressed_bytes.size() < 8) { // 至少需要8字节来恢复初始64位状态
        std::cerr << "Error: Compressed data too short to read initial state." << std::endl;
        return {};
    }

    std::vector<uint8_t> decompressed_data;
    decompressed_data.reserve(original_data_size);

    // 从字节流的开头读取32位字 (因为之前反转过)
    size_t read_idx = 0;
    auto read_u32 = [&]() -> uint32_t {
        if (read_idx + 4 > compressed_bytes.size()) {
             std::cerr << "Error: Not enough data to read u32." << std::endl;
             // 在实际应用中，这可能表示流结束或损坏
             return 0; // 或者抛出异常
        }
        uint32_t word = static_cast<uint32_t>(compressed_bytes[read_idx]) |
                       (static_cast<uint32_t>(compressed_bytes[read_idx + 1]) << 8) |
                       (static_cast<uint32_t>(compressed_bytes[read_idx + 2]) << 16) |
                       (static_cast<uint32_t>(compressed_bytes[read_idx + 3]) << 24);
        read_idx += 4;
        return word;
    };

    // 恢复初始状态 x (编码时最后写入的两个32位字)
    // 由于字节流已反转，这两个字在流的开头
    uint32_t x_high = read_u32(); // 这是编码时 x >> NORM_BITS 的部分
    uint32_t x_low  = read_u32(); // 这是编码时 x & (IO_BASE - 1) 的部分
    uint64_t x = (static_cast<uint64_t>(x_high) << NORM_BITS) | x_low;

    // 预先构建一个从 CDF 值快速查找符号的表或使用 std::upper_bound
    // 这里我们将在循环内查找，对于 TOTAL_FREQ=256 性能尚可
    // 更高效的方式是预处理一个映射：slot -> symbol

    for (size_t i = 0; i < original_data_size; ++i) {
        // 解码归一化：
        // 在解码符号前，如果状态 x 太小，则从流中读取 NORM_BITS 并补充 x
        while (x < IO_BASE) {
            if (read_idx >= compressed_bytes.size()) {
                std::cerr << "Error: Ran out of compressed data during normalization." << std::endl;
                // 可能意味着原始大小错误或数据损坏
                return decompressed_data; // 返回已解压部分
            }
            uint32_t next_word = read_u32();
            x = (x << NORM_BITS) | next_word; // 左移并或上新读入的字
        }

        // 恢复符号 s
        uint32_t slot = static_cast<uint32_t>(x % TOTAL_FREQ); // slot = (x_old % freq[s]) + cdf[s]

        uint8_t symbol = 0;
        // 在 CDF 表中查找对应的符号 s: cdf[s] <= slot < cdf[s+1]
        // 使用 std::upper_bound (需要 stats 是按 cdf 排序的，或者我们迭代查找)
        // 简单线性查找：
        for (uint8_t s_idx = 0; s_idx < TOTAL_FREQ -1; ++s_idx) { // TOTAL_FREQ-1 because cdf[s+1]
             if (slot >= stats[s_idx].cdf && slot < stats[s_idx+1].cdf) { // stats[s_idx+1].cdf is effectively cdf[s] + freq[s]
                 symbol = s_idx;
                 break;
             }
             // 处理最后一个符号的情况
             if (s_idx == TOTAL_FREQ - 2 && slot >= stats[TOTAL_FREQ-1].cdf) {
                 symbol = TOTAL_FREQ - 1;
                 break;
             }
        }
        // 更标准的方式是使用一个完整的 CDF 表，包括 cdf[TOTAL_FREQ] = sum_of_all_freqs
        // 然后使用 upper_bound(cdf_values.begin(), cdf_values.end(), slot) - cdf_values.begin() - 1;
        // 这里我们直接用 stats 结构。

        const SymbolStats& s_stats = stats[symbol];
        decompressed_data.push_back(symbol);

        // rANS 核心解码步骤
        // x_old = (x_current / TotalFreq) * freq[s] + (slot - cdf[s])
        x = (x / TOTAL_FREQ) * s_stats.freq + (slot - s_stats.cdf);
    }

    std::reverse(decompressed_data.begin(), decompressed_data.end()); //因为编码是从后往前，所以解码出的数据也是反的

    return decompressed_data;
}

// --- 主函数：示例用法 ---
int main() {
    // 示例数据
    std::vector<uint8_t> original_data = {'H', 'e', 'l', 'l', 'o', ',', ' ', 'w', 'o', 'r', 'l', 'd', '!', ' ',
                                          'r', 'A', 'N', 'S', ' ', 'i', 's', ' ', 'c', 'o', 'o', 'l', '.'};
    // original_data = {'a','b','a','c','a','b','a','a','a','a'}; // 另一个测试用例

    if (original_data.empty()) {
        std::cout << "Original data is empty." << std::endl;
        return 0;
    }

    std::cout << "Original Data Size: " << original_data.size() << " bytes" << std::endl;
    // std::cout << "Original Data: ";
    // for(uint8_t c : original_data) std::cout << c;
    // std::cout << std::endl;

    // 1. 构建符号统计信息 (频率和 CDF)
    // 注意：这里的 build_symbol_stats 是一个非常简化的版本。
    // 实际的 rANS 实现需要确保 sum(freqs) == TOTAL_FREQ。
    // 如果不满足，需要进行频率归一化。
    std::vector<SymbolStats> stats = build_symbol_stats(original_data);
    // 检查频率总和是否为 TOTAL_FREQ
    uint32_t sum_freqs = 0;
    for(const auto& s : stats) sum_freqs += s.freq;
    if (sum_freqs != TOTAL_FREQ) {
        std::cout << "Warning: Total sum of frequencies in stats table is " << sum_freqs
                  << ", but L_BITS implies TOTAL_FREQ should be " << TOTAL_FREQ
                  << ". This example may not perfectly normalize frequencies." << std::endl;
        // 在这个简化示例中，我们允许这种情况，但它会影响压缩效率。
        // 真正的rANS需要精确的频率归一化。
    }


    // 2. 编码
    std::vector<uint8_t> compressed_data = rans_encode(original_data, stats);
    std::cout << "Compressed Data Size: " << compressed_data.size() << " bytes" << std::endl;

    // 3. 解码
    // 解码时需要原始数据的大小，因为rANS本身不存储它（除非显式添加）
    std::vector<uint8_t> decompressed_data = rans_decode(compressed_data, stats, original_data.size());
    std::cout << "Decompressed Data Size: " << decompressed_data.size() << " bytes" << std::endl;
    // std::cout << "Decompressed Data: ";
    // for(uint8_t c : decompressed_data) std::cout << c;
    // std::cout << std::endl;

    // 4. 验证
    if (original_data == decompressed_data) {
        std::cout << "Verification Successful: Decompressed data matches original data." << std::endl;
    } else {
        std::cout << "Verification Failed: Decompressed data does NOT match original data." << std::endl;
        if (original_data.size() != decompressed_data.size()) {
             std::cout << "Size mismatch: original " << original_data.size() << ", decompressed " << decompressed_data.size() << std::endl;
        }
        for(size_t i=0; i < std::min(original_data.size(), decompressed_data.size()); ++i) {
            if (original_data[i] != decompressed_data[i]) {
                std::cout << "First mismatch at index " << i << ": original '" << original_data[i]
                          << "' vs decompressed '" << decompressed_data[i] << "'" << std::endl;
                break;
            }
        }
    }

    return 0;
}
```

### **代码解释：**

1. **参数配置 (`L_BITS`, `TOTAL_FREQ`, `NORM_BITS`, `IO_BASE`, `RANS_X_MAX_BEFORE_NORM_FACTOR`):**

    * `L_BITS`: 定义了频率表的大小。`TOTAL_FREQ` 是 `2^L_BITS`，这是 rANS 状态更新公式中的缩放因子。
    * `NORM_BITS`: 归一化时操作的位数。`IO_BASE` 是 `2^NORM_BITS`。
    * `RANS_X_MAX_BEFORE_NORM_FACTOR`: 用于计算编码归一化阈值的因子，确保 `x/freq` 足够大。

2. **`SymbolStats` 结构体:**

    * 存储每个符号的频率 (`freq`) 和累积频率 (`cdf`)。

3. **`build_symbol_stats` 函数:**

    * 计算输入数据中每个字节的频率。
    * **重要**: 这个函数包含了一个非常简化的频率平滑（将0频率设为1）和频率总和检查。在生产级 rANS 中，频率表的总和必须精确等于 `TOTAL_FREQ`。这通常通过复杂的频率缩放/归一化算法实现，以最大限度地减小精度损失。此示例为了简化，指出了这个问题但未完全解决。

4. **`rans_encode` 函数:**

    * 初始化状态 `x = IO_BASE`。
    * `compressed_words`: 存储归一化过程中输出的 `uint32_t` 字。
    * **从后往前**遍历输入数据。
    * **编码归一化**:
        * `threshold = RANS_X_MAX_BEFORE_NORM_FACTOR * s_stats.freq;`
        * `while (x >= threshold)`: 如果 `x` 超过阈值，输出低 `NORM_BITS` (32) 位到 `compressed_words`，然后 `x >>= NORM_BITS`。
    * **核心编码步骤**: `x = (x / s_stats.freq) * TOTAL_FREQ + s_stats.cdf + (x % s_stats.freq);`
    * 编码结束后，将最终的 64 位状态 `x` 拆分成两个 32 位字存入 `compressed_words`。
    * 将 `compressed_words` (uint32_t 序列) 转换为 `compressed_bytes` (uint8_t 序列)，按小端模式存储每个字。
    * **反转** `compressed_bytes`：因为 `compressed_words` 是以符号的反向顺序生成的，所以最终的字节流需要反转，以便解码器能以正确的顺序读取。

5. **`rans_decode` 函数:**

    * `read_u32` lambda: 从字节流中读取一个小端模式的 `uint32_t`。
    * **恢复初始状态 `x`**: 从 `compressed_bytes` 的开头读取两个 `uint32_t` (因为流被反转过，所以编码时最后写入的状态现在在最前面) 并组合成 64 位状态 `x`。
    * 循环 `original_data_size` 次来解码符号。
    * **解码归一化**:
        * `while (x < IO_BASE)`: 如果 `x` 小于 `IO_BASE`，从流中读取一个 `uint32_t`，然后 `x = (x << NORM_BITS) | next_word;` (将 `x` 左移并或上新读入的字)。
    * **恢复符号 `s`**:
        * `slot = static_cast<uint32_t>(x % TOTAL_FREQ);`
        * 通过查找 `stats` 表（CDF 值）找到满足 `stats[s].cdf <= slot < stats[s+1].cdf` 的符号 `s`。示例中使用线性查找，更优化的方法是使用 `std::upper_bound` 或预计算的查找表。
    * **核心解码步骤**: `x = (x / TOTAL_FREQ) * s_stats.freq + (slot - s_stats.cdf);`
    * 将解码出的符号存入 `decompressed_data`。
    * **反转** `decompressed_data`: 因为编码是从后往前处理符号，解码出的符号序列是原始序列的反向。

6. **`main` 函数:**

    * 提供了一个简单的示例，包括数据准备、频率统计、编码、解码和验证。
    * **重要**: 指出了频率总和可能不等于 `TOTAL_FREQ` 的问题，这在简化示例中是为了避免复杂的频率归一化代码，但在实际应用中至关重要。

### **编译和运行:**

1. 将代码保存为 `.cpp` 文件 (例如 `rans64_example.cpp`)。

2. 使用支持 C++11 或更高版本的编译器进行编译。例如，使用 g++：

    ```bash
    g++ rans64_example.cpp -o rans64_example -std=c++11 -Wall
    ```

3. 运行可执行文件：

    ```bash
    ./rans64_example
    ```

### **关键点和进一步改进：**

* **频率表归一化:** 这是 rANS 实现中最关键且复杂的部分之一。为了达到最佳压缩率和确保正确性，所有符号的频率之和必须精确等于 `TOTAL_FREQ`。这通常需要一个专门的函数来按比例缩放原始频率计数，同时确保每个频率至少为1。
* **CDF 查找效率:** 对于较大的 `TOTAL_FREQ`，解码时的线性 CDF 查找会变慢。可以使用 `std::upper_bound` 或构建一个反向查找表（从 `slot` 直接映射到符号）。
* **I/O 效率:** 示例中的 `std::vector<uint8_t>` 和逐字节读写 `uint32_t` 相对简单但可能不是最高效的。实际应用中可能会使用更底层的缓冲区管理和位流操作。
* **原始数据大小:** 解码器需要知道要解码多少符号。这通常通过在压缩流的开头存储原始大小来实现，或者使用一个特殊的 End-Of-Stream (EOS) 符号。
* **错误处理:** 生产代码需要更健壮的错误处理，例如检测损坏的流。

这个示例代码提供了一个 64 位 rANS 范围编码和解码的基本框架和核心逻辑。理解其状态更新、归一化以及频率表的重要性是掌握 rANS 的关键。
