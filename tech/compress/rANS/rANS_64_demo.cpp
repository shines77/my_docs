
#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cstdint>
#include <map>

// --- rANS 核心常量定义 ---

// 使用64位无符号整数作为rANS的核心状态
using Rans64Type = uint64_t;

// 概率模型的总频率所占的比特数。总频率 M = 1 << RANS_PROB_BITS。
// M 必须是2的幂，这样可以把耗时的取模和除法运算转换成高效的位运算。
// 12比特意味着总频率是4096，这在精度和性能之间取得了很好的平衡。
constexpr uint32_t RANS_PROB_BITS = 12;
constexpr uint32_t RANS_PROB_SCALE = 1 << RANS_PROB_BITS;

// rANS状态的归一化区间的下界。
// 当状态 x 低于这个值时，需要从输入流中读取数据以补充状态。
// 我们使用32位整数进行I/O，所以这个下界设为 2^32 比较合适。
// 状态 x 将始终被维持在 [RANS_L, +∞) 这个区间内（除了初始状态）。
constexpr Rans64Type RANS_L = 1ULL << 32;

// --- 数据结构定义 ---

// 存储单个符号的统计信息
struct SymbolStats {
    uint32_t freq;      // 符号的频率
    uint32_t cum_freq;  // 符号的累积频率 (所有在它之前的符号的频率之和)
};

// --- 编码器实现 ---

struct Rans64Encoder {
    Rans64Type x;                   // rANS 状态
    std::vector<uint32_t> stream;   // 压缩后的数据流 (以32位为单位)

    // 构造函数：初始化状态 x
    Rans64Encoder() : x(RANS_L) {}

    // 编码一个符号
    // sym: 待编码符号的统计信息 (频率和累积频率)
    void encode(const SymbolStats& sym) {
        // 归一化：确保状态 x 足够大，可以进行下一步的编码操作
        // 如果 x 太大，可能会导致 x_new 溢出64位。
        // `(RANS_L >> RANS_PROB_BITS) * sym.freq` 是一个阈值，保证信息不会丢失。
        // 当 x 大于等于这个阈值时，就将 x 的低32位写出到数据流。
        while (x >= (RANS_L >> RANS_PROB_BITS) * sym.freq) {
            stream.push_back(static_cast<uint32_t>(x & 0xFFFFFFFF));
            x >>= 32;
        }

        // rANS 核心编码公式
        // 这个公式将当前状态 x 和符号 sym 的信息结合，生成一个新的、更大的状态 x。
        x = (x / sym.freq) * RANS_PROB_SCALE + (x % sym.freq) + sym.cum_freq;
    }

    // 结束编码，将最终的状态 x 写入数据流
    void flush() {
        // 编码结束后，状态 x 中仍然包含未输出的信息。
        // 将剩余的64位状态值全部写入流中。
        stream.push_back(static_cast<uint32_t>(x & 0xFFFFFFFF));
        stream.push_back(static_cast<uint32_t>((x >> 32) & 0xFFFFFFFF));
    }
};

// --- 解码器实现 ---

struct Rans64Decoder {
    Rans64Type x;        // rANS 状态
    const uint32_t* ptr; // 指向压缩数据流的指针

    // 从数据流初始化解码器状态
    void init(const std::vector<uint32_t>& stream) {
        // 从流的末尾读取最终状态值来初始化解码器
        // 因为 flush 是最后写入的，所以这里要最先读出来。
        const uint32_t* end_ptr = stream.data() + stream.size();
        uint64_t high = *(end_ptr - 1);
        uint64_t low = *(end_ptr - 2);
        x = (high << 32) | low;
        ptr = end_ptr - 2; // 指针移动到已读取数据的前面
    }

    // 解码一个符号
    // stats_vec: 包含所有符号统计信息的向量
    // cum_freq_to_symbol: 从累积频率快速查找符号的映射表
    uint8_t decode(const std::vector<SymbolStats>& stats_vec, const std::vector<uint8_t>& cum_freq_to_symbol) {
        // 1. 从当前状态 x 的低位提取出累积频率信息
        uint32_t cf = x & (RANS_PROB_SCALE - 1);

        // 2. 通过累积频率查找对应的符号
        uint8_t s = cum_freq_to_symbol[cf];
        const SymbolStats& sym = stats_vec[s];

        // 3. rANS 核心解码公式
        // 这是编码公式的逆运算，用于恢复编码该符号之前的状态。
        x = sym.freq * (x >> RANS_PROB_BITS) + cf - sym.cum_freq;

        // 4. 归一化
        // 如果状态 x 变得太小，就从数据流中读入32位来补充它。
        while (x < RANS_L) {
            x = (x << 32) | (*--ptr);
        }

        return s;
    }
};


// --- 主函数和辅助函数 ---

// 从数据构建频率模型
void build_stats(const std::vector<uint8_t>& data, std::vector<SymbolStats>& stats_vec, std::vector<uint8_t>& cum_freq_to_symbol) {
    // 1. 统计每个字节出现的次数
    std::map<uint8_t, uint32_t> counts;
    for (uint8_t byte : data) {
        counts[byte]++;
    }

    // 2. 将原始计数值按比例缩放到总和为 RANS_PROB_SCALE
    uint64_t total_count = data.size();
    uint32_t current_sum = 0;

    // 为了避免浮点数运算，我们使用整数乘法和除法来进行缩放
    for (auto const& [symbol, count] : counts) {
        uint32_t scaled_freq = (static_cast<uint64_t>(count) * RANS_PROB_SCALE) / total_count;
        if (scaled_freq == 0) scaled_freq = 1; // 确保每个出现的符号频率至少为1
        stats_vec[symbol].freq = scaled_freq;
        current_sum += scaled_freq;
    }

    // 3. 调整频率，使其总和精确等于 RANS_PROB_SCALE
    // 这是一个简单的调整策略，将多出或不足的部分加到频率最高的符号上
    int32_t error = current_sum - RANS_PROB_SCALE;
    if (error != 0) {
        uint8_t best_s = 0;
        uint32_t max_freq = 0;
        for (int i = 0; i < 256; ++i) {
            if (stats_vec[i].freq > max_freq) {
                max_freq = stats_vec[i].freq;
                best_s = i;
            }
        }
        stats_vec[best_s].freq -= error;
    }

    // 4. 计算累积频率
    uint32_t cum_freq = 0;
    for (int i = 0; i < 256; ++i) {
        if (stats_vec[i].freq > 0) {
            stats_vec[i].cum_freq = cum_freq;
            cum_freq += stats_vec[i].freq;
        }
    }

    // 5. 创建从累积频率到符号的快速查找表
    cum_freq_to_symbol.resize(RANS_PROB_SCALE);
    for (int i = 0; i < 256; ++i) {
        if (stats_vec[i].freq > 0) {
            for (uint32_t j = stats_vec[i].cum_freq; j < stats_vec[i].cum_freq + stats_vec[i].freq; ++j) {
                cum_freq_to_symbol[j] = i;
            }
        }
    }
}

int main()
{
    // 1. 准备原始数据
    std::string original_str = "This is a simple example for rANS encoding and decoding. It is a very efficient entropy coder.";
    std::vector<uint8_t> original_data(original_str.begin(), original_str.end());
    std::cout << "原始数据大小: " << original_data.size() << " 字节" << std::endl;

    // 2. 构建频率统计模型
    std::vector<SymbolStats> stats_vec(256, {0, 0});
    std::vector<uint8_t> cum_freq_to_symbol;
    build_stats(original_data, stats_vec, cum_freq_to_symbol);

    // 3. 编码过程
    Rans64Encoder encoder;
    // rANS 需要从后向前编码数据
    for (auto it = original_data.rbegin(); it != original_data.rend(); ++it) {
        encoder.encode(stats_vec[*it]);
    }
    encoder.flush();
    std::vector<uint32_t> compressed_stream = encoder.stream;
    // 因为编码时是 push_back，但逻辑上是写到流的“开头”，所以这里反转一下方便理解
    std::reverse(compressed_stream.begin(), compressed_stream.end());

    std::cout << "压缩后数据大小: " << compressed_stream.size() * sizeof(uint32_t) << " 字节" << std::endl;
    double ratio = (double)(compressed_stream.size() * sizeof(uint32_t)) / original_data.size();
    std::cout << "压缩率: " << ratio * 100.0 << "%" << std::endl;


    // 4. 解码过程
    Rans64Decoder decoder;
    decoder.init(compressed_stream);
    std::vector<uint8_t> decoded_data(original_data.size());
    // 解码是正向进行的
    for (size_t i = 0; i < original_data.size(); ++i) {
        decoded_data[i] = decoder.decode(stats_vec, cum_freq_to_symbol);
    }

    // 5. 验证结果
    if (original_data == decoded_data) {
        std::cout << "成功: 解码后的数据与原始数据完全一致！" << std::endl;
        std::cout << "解码后的字符串: " << std::string(decoded_data.begin(), decoded_data.end()) << std::endl;
    } else {
        std::cout << "失败: 解码数据与原始数据不匹配。" << std::endl;
    }

    return 0;
}
