# 64位 rANS 压缩算法的原理和实现

## 问题

压缩算法中的 rANS 范围编码的压缩、解压原理和具体实现，请使用 64 位整型版本，使用 C++ 代码示例，并添加中文注释。

## 回答

（来自 gemini-2.5-flash）

好的，我们来详细讲解压缩算法中的 rANS（ranged Asymmetric Numeral Systems）范围编码的原理、64 位整型实现以及 C++ 代码示例。

### **1. rANS 范围编码原理**

rANS 是一种先进的熵编码方法，它属于 ANS（Asymmetric Numeral Systems）家族。与传统的算术编码类似，rANS 根据符号的概率（频率）来分配编码空间，实现接近理论极限的压缩率。但与算术编码维护一个浮动范围不同，rANS 使用一个整数状态变量 `x` 来表示当前的编码状态。

核心思想：

rANS 的编码和解码过程是一对互逆的运算。它通过将一个旧状态 `x_old` 和一个符号 `s` 映射到一个新状态 `x_new` 来实现编码。解码则是反过来，从当前状态 `x_current` 中恢复出符号 `s` 和旧状态 `x_old`。

假设我们有一个符号集，总频率为 `TotalFreq` (通常是 2 的幂，如 256 或 4096)。每个符号 `s` 有其频率 `freq[s]` 和累积频率 `cdf[s]` (即频率小于 `s` 的所有符号的频率之和)。

* **编码 (Encoding):**

    编码过程通常是**从后往前**处理输入符号。对于要编码的当前符号 `s`，其频率为 `freq`，累积频率为 `cdf`。当前的编码状态为 `x`。编码操作将 `x` 更新为 `x_new`：

    `x_new = (x / freq) * TotalFreq + cdf + (x % freq)`

    这个公式可以理解为：将状态 `x` 分解为 `x / freq` (商) 和 `x % freq` (余数)。新的状态 `x_new` 是将商乘以总频率 `TotalFreq`，再加上符号 `s` 的累积频率 `cdf`，最后再加上余数 `x % freq`。这巧妙地将符号 `s` 的信息（通过 `cdf` 和 `freq` 体现）嵌入到新的状态 `x_new` 中。

    为了防止状态 `x` 过大导致溢出，或者过小导致精度损失，编码过程中需要进行**归一化 (Normalization)**。当状态 `x` 达到一个预设的上限阈值时，我们会将 `x` 的低位部分（通常是 `NORM_BITS` 位）输出到压缩流中，并将 `x` 右移 `NORM_BITS` 位。这保证了 `x` 始终在一个可管理的范围内。这个上限阈值的设计与 `NORM_BITS` 和 `TotalFreq` 相关，通常是为了保证 `x / freq` 至少大于一个特定值，以便在更新后状态能保持在解码所需的范围内。

    编码的输出流是反向的：先处理的符号，其对应的归一化输出位会出现在压缩流的后面。因此，编码器通常将输出位存储在一个缓冲区中，最后再反转缓冲区写入文件。

* **解码 (Decoding):**

    解码过程是**从前往后**处理压缩流。我们从压缩流的末尾读取初始状态 `x`。对于当前状态 `x`，我们需要恢复出符号 `s` 和旧状态 `x_old`。

    1. **恢复符号 s:** 回顾编码公式 `x_current = (x_old / freq[s]) * TotalFreq + cdf[s] + (x_old % freq[s])`。令 `q = x_old / freq[s]` 和 `r = cdf[s] + (x_old % freq[s])`。则 `x_current = q * TotalFreq + r`。这意味着 `q = x_current / TotalFreq` (整数除法) 且 `r = x_current % TotalFreq` (取余)。
        关键在于 `r = cdf[s] + (x_old % freq[s])`。由于 `0 <= (x_old % freq[s]) < freq[s]`，所以 `cdf[s] <= r < cdf[s] + freq[s]`。因此，通过计算 `slot = x_current % TotalFreq`，我们只需找到满足 `cdf[s] <= slot < cdf[s] + freq[s]` 的符号 `s` 即可。这是一个基于 CDF 表的查找操作。

    2. **恢复旧状态 x_old:** 一旦找到符号 `s` (及其 `freq[s]` 和 `cdf[s]`)，我们就可以计算 `q = x_current / TotalFreq`。我们知道 `x_old = q * freq[s] + (x_old % freq[s])`。从 `r = cdf[s] + (x_old % freq[s])`，我们得到 `(x_old % freq[s]) = r - cdf[s]`。因此，`x_old = (x_current / TotalFreq) * freq[s] + (x_current % TotalFreq) - cdf[s]`。这就是解码的核心状态更新公式。

    为了防止状态 `x` 过小（低于预设的下限阈值），解码过程中也需要进行归一化。当下状态 `x` 低于阈值时，我们会从压缩流中读取 `NORM_BITS` 位，并用这些位来“补充”状态 `x`，即将 `x` 左移 `NORM_BITS` 位并加上读取的位。这保证了 `x` 始终在解码所需的范围内。这个下限阈值通常是 `1 << NORM_BITS`。

总的来说，rANS 编码器通过将信息“推入”状态的高位，并在状态溢出时将高位输出；解码器则从状态的低位“拉出”信息（符号），并在状态不足时从输入流补充低位。

### **2. 64 位整型实现**

使用 64 位整型 (`uint64_t`) 作为状态变量 `x` 可以处理更大的范围，提高精度和潜在的压缩效率。关键参数选择：

* **状态类型:** `uint64_t`
* **`L_BITS`:** 用于表示总频率的位数。例如，如果处理字节数据，`L_BITS = 8`，`TotalFreq = 1 << 8 = 256`。如果使用更大的上下文或更高精度的频率，可以使用 `L_BITS = 12`，`TotalFreq = 1 << 12 = 4096`。这里我们先以字节为单位，使用 `L_BITS = 8`，`TotalFreq = 256`。
* **`NORM_BITS`:** 每次归一化处理的位数。对于 64 位状态，选择 `NORM_BITS = 32` 是一个常见的且高效的选择，因为它与 32 位字长匹配，方便 I/O。
* **`IO_BASE`:** `1ULL << NORM_BITS`，即 2 的 `NORM_BITS` 次幂。这是归一化时每次输出/输入的单位。对于 `NORM_BITS = 32`，`IO_BASE = 1ULL << 32`。
* **状态范围:** 编码时，状态 `x` 需要足够大，以保证 `(x / freq) * TotalFreq` 这一项的计算精度。一个常见的编码归一化阈值是 `1ULL << (64 - NORM_BITS)`。当 `x` 大于等于这个阈值时，输出 `NORM_BITS` 位。解码时，状态 `x` 需要至少为 `IO_BASE` (`1ULL << NORM_BITS`) 来进行符号查找和更新。当 `x` 小于 `IO_BASE` 时，需要读取 `NORM_BITS` 位进行补充。

**编码归一化阈值:**

编码时，状态 `x` 在更新前需要满足 `x >= (IO_BASE / TotalFreq) * freq[s]`，以确保 `x / freq[s]` 足够大。如果 `x` 小于这个值，就需要先输出位并右移 `x` 来放大 `x`。这个阈值可以写成 `(1ULL << (NORM_BITS - L_BITS)) * freq[s]`。

然而，更常见的实现方式是让编码器在 `x` 达到一个更高的阈值（如 `1ULL << (64 - NORM_BITS)`）时输出位，这样可以简化逻辑。我们采用后一种更简洁的方式：编码时，当 `x >= (1ULL << (64 - NORM_BITS))` 时，输出 `NORM_BITS` 位并右移 `x`。

### **3. C++ 代码实现**

下面是一个简单的 64 位 rANS 范围编码和解码的 C++ 实现示例。为了简化，我们使用固定的字节频率表 (TotalFreq = 256)，并且 I/O 直接使用 `std::vector<uint8_t>` 作为缓冲区。

```cpp
#include <iostream>
#include <vector>
#include <numeric> // for std::partial_sum
#include <algorithm> // for std::reverse, std::upper_bound
#include <cstdint>   // for uint64_t, uint32_t, uint8_t

// rANS 参数
const int L_BITS = 8; // 频率表位数，对于字节就是 8
const uint32_t TotalFreq = 1 << L_BITS; // 总频率，对于字节是 256

const int NORM_BITS = 32; // 每次归一化处理的位数
const uint64_t IO_BASE = 1ULL << NORM_BITS; // 归一化时的基数 (2^32)
const uint64_t RANGE_BITS = 64; // 状态变量的位数

// 编码归一化阈值：当状态达到此值时，输出 NORM_BITS 位
// 这是一个关键参数，决定了状态需要保持的最小值，以保证 (x / freq) * TotalFreq 不损失过多精度
// 更准确的阈值计算应考虑 freq，例如 (IO_BASE / TotalFreq) * freq
// 但为了简化，我们使用一个与 NORM_BITS 相关的阈值，确保右移后状态仍在 IO_BASE 或以上
// 这里使用 1ULL << (RANGE_BITS - NORM_BITS)，确保右移 NORM_BITS 后，状态至少为 1ULL << (RANGE_BITS - 2*NORM_BITS)
// 同时，编码前的状态 x 必须 >= (IO_BASE / TotalFreq) * freq[s] 才能保证编码后状态 >= IO_BASE
// 我们采用一种常见的实现方式：当 x >= (1ULL << (RANGE_BITS - NORM_BITS)) 时输出位
const uint64_t ENC_NORM_THRESH = 1ULL << (RANGE_BITS - NORM_BITS); // 1ULL << 32

// 解码归一化阈值：当状态低于此值时，读取 NORM_BITS 位
const uint64_t DEC_NORM_THRESH = IO_BASE; // 1ULL << 32

// 计算频率表
std::vector<uint32_t> calculate_frequencies(const std::vector<uint8_t>& data) {
    std::vector<uint32_t> freq(TotalFreq, 0);
    for (uint8_t byte : data) {
        freq[byte]++;
    }
    // 保证所有频率都至少为 1，避免除以零和 CDF 问题
    // 实际应用中需要更复杂的平滑处理
    for (uint32_t& f : freq) {
        if (f == 0) f = 1;
    }
    return freq;
}

// 构建 CDF 表 (Cumulative Distribution Function)
// cdf[i] = sum(freq[0]...freq[i-1])
// cdf 的大小比 freq 大 1，cdf[TotalFreq] 等于 TotalFreq 的总和
std::vector<uint32_t> build_cdf(const std::vector<uint32_t>& freq) {
    std::vector<uint32_t> cdf(TotalFreq + 1, 0);
    std::partial_sum(freq.begin(), freq.end(), cdf.begin() + 1);
    // 检查 CDF 表的最后一个值是否等于总频率（所有频率之和）
    // std::accumulate(freq.begin(), freq.end(), 0U)
    if (cdf[TotalFreq] != std::accumulate(freq.begin(), freq.end(), 0U)) {
         std::cerr << "Error: CDF total does not match sum of frequencies." << std::endl;
    }
    return cdf;
}

// rANS 编码
std::vector<uint8_t> rans_encode(const std::vector<uint8_t>& data, const std::vector<uint32_t>& freq, const std::vector<uint32_t>& cdf) {
    std::vector<uint8_t> compressed_data;
    compressed_data.reserve(data.size()); // 预留空间

    // 输出 NORM_BITS 位到缓冲区
    auto write_norm_bits = [&](uint32_t bits) {
        // 将 uint32_t 拆成 4 个 uint8_t
        compressed_data.push_back(static_cast<uint8_t>(bits & 0xFF));
        compressed_data.push_back(static_cast<uint8_t>((bits >> 8) & 0xFF));
        compressed_data.push_back(static_cast<uint8_t>((bits >> 16) & 0xFF));
        compressed_data.push_back(static_cast<uint8_t>((bits >> 24) & 0xFF));
    };

    // 初始状态，通常设置为 IO_BASE
    uint64_t x = IO_BASE;

    // 从后往前编码数据
    for (auto it = data.rbegin(); it != data.rend(); ++it) {
        uint8_t symbol = *it;
        uint32_t current_freq = freq[symbol];
        uint32_t current_cdf = cdf[symbol];

        // 编码归一化：当状态 x 达到上限阈值时，输出 NORM_BITS 位并右移
        // 这里的阈值确保 x / current_freq >= (ENC_NORM_THRESH / TotalFreq) 大致成立
        // 使得更新后的状态 x 足够大
        uint64_t limit = (ENC_NORM_THRESH / TotalFreq) * current_freq;
        if (x >= limit) {
             // 为了更精确和避免除法，可以使用位移计算阈值
             // uint64_t limit_shifted = (1ULL << (RANGE_BITS - L_BITS - NORM_BITS));
             // while (x >= limit_shifted * current_freq) {
             //    write_norm_bits(x & (IO_BASE - 1));
             //    x >>= NORM_BITS;
             // }
            write_norm_bits(x & (IO_BASE - 1));
            x >>= NORM_BITS;
        }


        // rANS 编码状态更新
        x = (x / current_freq) * TotalFreq + current_cdf + (x % current_freq);

        // 另一种常见的更新公式 (tANS 变体)
        // x = ((x / current_freq) << L_BITS) + current_cdf + (x % current_freq);
        // 这里我们使用更通用的 TotalFreq
    }

    // 编码完成后，输出最终状态 x (64 位)
    for (int i = 0; i < 8; ++i) {
        compressed_data.push_back(static_cast<uint8_t>((x >> (i * 8)) & 0xFF));
    }

    // 编码输出是反序的，需要反转整个缓冲区
    std::reverse(compressed_data.begin(), compressed_data.end());

    return compressed_data;
}

// rANS 解码
std::vector<uint8_t> rans_decode(const std::vector<uint8_t>& compressed_data, const std::vector<uint32_t>& freq, const std::vector<uint32_t>& cdf) {
    std::vector<uint8_t> decompressed_data;
    if (compressed_data.size() < 8) {
        // 数据太短，无法读取初始状态
        return decompressed_data;
    }

    // 读取 NORM_BITS 位
    size_t read_pos = 0;
    auto read_norm_bits = [&]() -> uint32_t {
        if (read_pos + 4 > compressed_data.size() - 8) {
            // 没有足够的 NORM_BITS (4 bytes) 可读，这通常意味着输入流已耗尽
            // 在实际应用中需要更健壮的错误处理
             std::cerr << "Error: Ran out of bits during decoding normalization." << std::endl;
             return 0; // 返回 0 或采取其他错误处理
        }
        uint32_t bits = 0;
        bits |= static_cast<uint32_t>(compressed_data[read_pos++]);
        bits |= static_cast<uint32_t>(compressed_data[read_pos++]) << 8;
        bits |= static_cast<uint32_t>(compressed_data[read_pos++]) << 16;
        bits |= static_cast<uint32_t>(compressed_data[read_pos++]) << 24;
        return bits;
    };

    // 读取初始状态 x (在编码输出的末尾，但在压缩流的开头读取)
    uint64_t x = 0;
    size_t state_start_pos = compressed_data.size() - 8;
    if (state_start_pos < 0) {
         std::cerr << "Error: Compressed data too short to read initial state." << std::endl;
         return decompressed_data;
    }
    for (int i = 0; i < 8; ++i) {
        x |= static_cast<uint64_t>(compressed_data[state_start_pos + i]) << (i * 8);
    }

    // 从前往后解码符号
    while (read_pos < state_start_pos) { // 当还有归一化输出位可读时循环
        // 解码归一化：当状态 x 低于下限阈值时，读取 NORM_BITS 位并左移补充
        while (x < DEC_NORM_THRESH) {
            uint32_t bits = read_norm_bits();
            x = x * IO_BASE + bits;
        }

        // 恢复符号 s
        uint32_t slot = x % TotalFreq; // (x_old % freq[s]) + cdf[s]

        // 在 CDF 表中查找对应的符号 s
        // cdf[s] <= slot < cdf[s+1]
        uint8_t symbol = 0;
        // 可以使用 std::upper_bound 进行二分查找，更高效
        auto it = std::upper_bound(cdf.begin(), cdf.end(), slot);
        symbol = static_cast<uint8_t>(std::distance(cdf.begin(), it) - 1);

        // 或者线性查找 (对于 TotalFreq=256 足够快)
        // while (slot >= cdf[symbol + 1]) {
        //     symbol++;
        // }


        decompressed_data.push_back(symbol);

        // rANS 解码状态更新
        uint32_t current_freq = freq[symbol];
        uint32_t current_cdf = cdf[symbol];
        x = (x / TotalFreq) * current_freq + slot - current_cdf;
    }

    // 最后的状态 x 应该回到初始的 IO_BASE，但在实际应用中，可能因为填充等原因略有不同
    // 如果需要解码出原始数据的大小，可以在压缩流中存储大小信息

    return decompressed_data;
}


int main() {
    // 示例数据
    std::vector<uint8_t> original_data = {'a', 'a', 'a', 'b', 'b', 'c', 'a', 'b', 'c', 'c', 'c', 'c', 'a', 'a'};
    // std::vector<uint8_t> original_data = { 'H', 'e', 'l', 'l', 'o', ' ', 'W', 'o', 'r', 'l', 'd', '!' };

    std::cout << "Original Data Size: " << original_data.size() << " bytes" << std::endl;

    // 1. 计算频率和 CDF
    std::vector<uint32_t> freq = calculate_frequencies(original_data);
    std::vector<uint32_t> cdf = build_cdf(freq);

    // 打印频率和 CDF (可选)
    // std::cout << "Frequencies:" << std::endl;
    // for(int i = 0; i < TotalFreq; ++i) {
    //     if (freq[i] > 0) {
    //         std::cout << (char)i << ": " << freq[i] << " (cdf: " << cdf[i] << ")" << std::endl;
    //     }
    // }
    // std::cout << "CDF[" << TotalFreq << "]: " << cdf[TotalFreq] << std::endl;


    // 2. 编码
    std::vector<uint8_t> compressed_data = rans_encode(original_data, freq, cdf);
    std::cout << "Compressed Data Size: " << compressed_data.size() << " bytes" << std::endl;

    // 3. 解码
    std::vector<uint8_t> decompressed_data = rans_decode(compressed_data, freq, cdf);
    std::cout << "Decompressed Data Size: " << decompressed_data.size() << " bytes" << std::endl;

    // 4. 验证
    if (original_data == decompressed_data) {
        std::cout << "Verification Successful: Decompressed data matches original data." << std::endl;
    } else {
        std::cout << "Verification Failed: Decompressed data does NOT match original data." << std::endl;
        // 打印部分数据进行对比 (可选)
        // std::cout << "Original: ";
        // for(int i = 0; i < std::min(original_data.size(), (size_t)20); ++i) std::cout << original_data[i];
        // std::cout << std::endl;
        // std::cout << "Decompressed: ";
        // for(int i = 0; i < std::min(decompressed_data.size(), (size_t)20); ++i) std::cout << decompressed_data[i];
        // std::cout << std::endl;
    }

    return 0;
}
```

### **代码解释：**

1. **常量定义:** 定义了 `L_BITS`, `TotalFreq`, `NORM_BITS`, `IO_BASE`, `ENC_NORM_THRESH`, `DEC_NORM_THRESH` 等关键参数。
2. **`calculate_frequencies`:** 计算输入数据中每个字节的频率。为了简单和防止零频率，这里将所有零频率的符号频率设置为 1。实际应用中需要更复杂的统计建模和平滑技术。
3. **`build_cdf`:** 根据频率表构建累积频率表。`cdf[i]` 存储前 `i` 个符号的总频率。
4. **`rans_encode`:**
    * 初始化状态 `x` 为 `IO_BASE`。
    * 创建一个 `compressed_data` 向量用于存储输出字节。
    * `write_norm_bits` lambda 函数用于将 32 位的 `uint32_t` 拆分成 4 个字节并添加到 `compressed_data`。
    * 从输入数据的**末尾**开始循环处理每个符号。
    * 在处理符号前，进行编码归一化：如果状态 `x` 大于等于 `ENC_NORM_THRESH`，则将 `x` 的低 `NORM_BITS` 位（即 `x & (IO_BASE - 1)`）输出到缓冲区，并将 `x` 右移 `NORM_BITS` 位。这个过程会重复直到 `x` 小于阈值。*注意：代码中的简单阈值检查 `x >= limit` 是一个近似，更精确的实现会根据 `freq` 动态调整 `limit`。这里使用 `ENC_NORM_THRESH` 作为基础阈值是因为它足够简单且常见，但不是最严格的理论实现。*
    * 执行 rANS 状态更新公式 `x = (x / current_freq) * TotalFreq + current_cdf + (x % current_freq)`。
    * 循环结束后，将最终的状态 `x` (64 位) 追加到缓冲区。
    * 因为编码输出是反向的，所以最后需要反转整个 `compressed_data` 向量。
5. **`rans_decode`:**
    * 创建一个 `decompressed_data` 向量用于存储解压后的字节。
    * `read_norm_bits` lambda 函数用于从 `compressed_data` 中按顺序读取 4 个字节并组合成一个 `uint32_t`。
    * 首先从 `compressed_data` 的**末尾**读取初始状态 `x` (8 字节)。
    * 进入解码循环，循环条件是还有归一化输出位可读 (`read_pos < state_start_pos`)。
    * 在解码符号前，进行解码归一化：如果状态 `x` 小于 `DEC_NORM_THRESH`，则从输入流读取 `NORM_BITS` 位，将 `x` 左移 `NORM_BITS` 位，并加上读取的位。这个过程会重复直到 `x` 大于等于 `DEC_NORM_THRESH`。
    * 计算 `slot = x % TotalFreq`。
    * 在 CDF 表中查找满足 `cdf[s] <= slot < cdf[s+1]` 的符号 `s`。示例中使用了 `std::upper_bound` 进行二分查找，并计算符号索引。线性查找也是一个简单替代方案。
    * 将恢复的符号添加到 `decompressed_data`。
    * 执行 rANS 解码状态更新公式 `x = (x / TotalFreq) * current_freq + slot - current_cdf`。
    * 循环结束后，返回 `decompressed_data`。
6. **`main`:** 包含一个简单的示例，演示了如何使用这些函数进行压缩和解压，并验证结果。

### **如何编译和运行：**

1. 将代码保存为 `.cpp` 文件 (例如 `rans64.cpp`)。

2. 使用支持 C++11 或更高版本的编译器进行编译。例如，使用 g++：

    ```bash
    g++ rans64.cpp -o rans64 -std=c++11 -Wall
    ```

3. 运行可执行文件：

    ```bash
    ./rans64
    ```

输出应该显示原始数据大小、压缩数据大小、解压数据大小以及验证结果。对于示例数据，压缩后的数据大小应该小于原始数据大小。

### **进一步改进和注意事项：**

* **频率建模:** 简单的频率统计并不能处理所有情况（如新符号）。实际应用中需要更复杂的频率建模，例如基于上下文的频率、自适应频率或使用固定的伪计数。
* **零频率处理:** 代码中简单地将零频率设置为 1。更好的方法是在构建 CDF 时为所有可能的符号分配一个最小的频率，或者使用逃逸机制处理未见过的符号。
* **I/O 效率:** 示例中的 `std::vector<uint8_t>` 读写比较简单，但效率不高。高性能实现会使用更底层的字节流或位流操作。
* **错误处理:** 示例中的错误处理非常基础。实际应用需要处理文件 I/O 错误、数据损坏等情况。
* **数据大小:** 压缩流中通常需要包含原始数据的大小，以便解码器知道何时停止。
* **CDF 查找:** 对于较大的 `TotalFreq` (如 4096)，使用 `std::upper_bound` 进行二分查找比线性查找效率更高。
* **编码阈值:** 编码归一化阈值 `ENC_NORM_THRESH` 的选择对性能和压缩率有影响。更精细的实现会根据当前符号的频率动态调整阈值。代码中使用的 `(1ULL << (RANGE_BITS - NORM_BITS))` 是一个常见的简化阈值。
* **rANS 变体:** rANS 有多种变体，如 tANS (tabled ANS)。这个实现更接近于标准 rANS 的描述。

这个示例提供了一个 64 位 rANS 范围编码和解码的基本框架和核心逻辑。理解其状态更新和归一化原理是掌握 rANS 的关键。
