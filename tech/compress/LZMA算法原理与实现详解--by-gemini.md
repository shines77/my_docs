# **LZMA无损压缩算法详解**

## 概述

* 来自：[Gemini 2.5 Pro](https://gemini.google.com/)
* 网址：[Gemini Deep Research](https://gemini.google.com/app/397e23ee9b71af48?hl=zh-cn)
* Google Docs：[Google 文档](https://docs.google.com/document/d/1GT3FKnR9rUsPYJgc7B_ZMc_hTSnWkU8beVbGxAx5soY/edit?tab=t.0)

## **1. 引言**

无损压缩技术在数据存储和传输领域扮演着至关重要的角色。它能够在不丢失任何原始信息的前提下减小数据体积，从而提高存储效率、降低带宽需求。在众多的无损压缩算法中，LZMA（Lempel-Ziv-Markov chain Algorithm）凭借其卓越的压缩率和相对较快的解压缩速度，成为了广泛应用的一种算法。LZMA 由伊戈尔·帕夫洛夫于 1998 年开发，并于 2001 年集成到著名的 7-Zip 压缩软件中 [1]。随后，它也被广泛应用于 xz utils 等开源工具中 [2]。LZMA 算法及其相关的 LZMA SDK 均以开源形式发布 [1]，这极大地促进了其在各种应用场景中的普及。

LZMA算法的主要特点和优势包括：极高的压缩率，通常优于gzip和bzip2等传统压缩算法，能够显著减小文件体积，使得7z格式的压缩包在许多情况下都更小 3；快速的解压缩速度，这对于高效的数据检索至关重要，其解压缩速度远超bzip2 3；可变的字典大小，允许根据输入数据的特性进行优化，7-Zip中最大可达4GB 6，早期版本和SDK中最大为1GB [3]，Snippet [4] 提到新版本中字典大小可达4GB。此外，LZMA算法由于其较小的解压缩内存需求（取决于字典大小）和代码体积（约为5KB） 3，非常适合应用于嵌入式系统。部分实现，如LZMA2，还支持多线程和P4的超线程技术，以进一步提升性能 6。XZ Utils中的LZMA2提供了更好的多线程支持 [6]。

研究表明，LZMA之所以能够实现如此高的压缩率和相对较快的解压缩速度，得益于其复杂而精妙的设计。可变的字典大小为算法提供了根据不同数据特征进行调整的灵活性，从而在压缩效率和资源消耗之间取得平衡。其在解压缩方面的优势确保了用户能够快速地访问压缩数据，这对于提升用户体验至关重要。LZMA在嵌入式领域的适用性表明其在资源受限的环境中也能够发挥重要作用。多线程技术的支持则使其能够充分利用现代多核处理器的计算能力，进一步提升性能。

## **2. LZMA的基本原理**

LZMA算法是基于著名的LZ77算法改进而来的一种压缩方法 [3]。LZ77 算法的核心思想是利用滑动窗口来寻找数据中的重复模式。LZMA在此基础上进行了优化和扩展，主要体现在以下几个方面：

### **2.1 滑动窗口**

LZMA 使用一个称为滑动窗口（或字典、历史缓冲区）的概念来记录最近出现的数据 9。这个窗口的大小是可变的 3，最大可以达到 4 GB（在 7-Zip 中） 6，更早的版本和SDK中则为 1GB [3]。Snippet [4] 指出，在 21.03 beta 版本之后，字典大小可以达到4GB。Snippet [13]表明，字典大小可以通过 `-d` 参数进行设置。Snippet [25] 讨论了 .lzma 文件头中字典大小的存储方式，而 [18] 则说明最小字典大小为 4 KB。滑动窗口在概念上被划分为两个部分：搜索缓冲区（当前的字典）和前瞻缓冲区（待编码的数据）[1]。

### **2.2 匹配查找**

编码过程中，LZMA算法会在搜索缓冲区中查找与前瞻缓冲区起始部分的最长字节序列相匹配的子串 1。为了高效地完成匹配查找，LZMA通常会使用哈希表 1 或二叉树 15 等数据结构。Snippet [15] 提到，可以通过mf参数选择匹配查找算法，包括 btMode（二叉树）和 hc\*（哈希链）。Snippet [13] 中的 fb（快速字节数）和 mc（匹配查找周期数）参数也会影响匹配过程，它们控制着压缩速度和压缩率之间的权衡。

### **2.3 编码匹配信息**

如果在搜索缓冲区中找到了与前瞻缓冲区中的数据相匹配的子串，LZMA会将这个匹配信息编码为一对数字：距离（distance或offset），表示匹配子串在滑动窗口中相对于当前位置的偏移量；长度（length），表示匹配子串的字节数 [1]。为了进一步提高编码效率，LZMA会维护一个最近使用的四个距离的历史记录 1。如果后续的匹配距离与历史记录中的某个距离相同，则可以使用更短的2位编码来表示 [1]。

### **2.4 处理非匹配字节**

如果在搜索缓冲区中没有找到匹配的子串，或者找到的匹配长度小于某个阈值（通常为2个字节 12），那么前瞻缓冲区中的第一个字节将被视为一个字面值（literal），并直接进行编码 [1]。

LZMA算法的核心在于其能够有效地利用滑动窗口机制来发现并编码数据中的重复序列。通过可变的窗口大小，算法可以适应不同冗余程度的数据。高效的匹配查找机制对于保证压缩速度至关重要。对匹配信息（距离和长度）以及非匹配字节（字面值）的区分和编码是 LZ77 阶段的主要输出。

## **3. LZMA编码规范和字节流格式**

LZMA的编码规范定义了压缩数据的组织方式，包括文件头和压缩数据的具体结构。对于.lzma文件格式，其结构如下 [17]：

### **3.1 属性字节 (1 字节)**

该字节包含编码模型所需的参数：字面值上下文位数（lc，3位，取值范围\[0-8\]）、字面值位置位数（lp，2位，取值范围\[0-4\]）和位置位数（pb，3位，取值范围\[0-4\]） 17。这些参数通过公式 Properties \= (pb \* 5 \+ lp) \* 9 \+ lc 进行编码 18。例如，Snippet 26中提到，值 0x5D 对应于 lc=3, lp=0, pb=2。Snippet [18] 还指出，对于使用 LZMA 的新格式，建议满足约束条件 lc \+ lp \<= 4。

### **3.2 字典大小 (4 字节)**

字典大小以 32 位无符号整数的形式存储，采用小端字节序 [17]。Snippet [26] 给出了一个例子：0x0080\_0000（8MB）。为了获得最佳的兼容性，推荐使用2的幂或者2 ^ n \+ 2 ^ (n-1) 的大小 25。最小字典大小为 4 KB（212字节）[18]。如果属性字节中隐含的字典大小小于4KB，解码器应使用 4 KB [18]。

### **3.3 未压缩大小 (8 字节)**

原始数据的未压缩大小以 64 位无符号整数的形式存储，同样采用小端字节序 [17]。一个特殊的值0xFFFF\_FFFF\_FFFF\_FFFF表示未压缩大小未知，此时压缩数据流中会包含一个流结束（EOS）标记来指示解码的结束 [17]。Snippet 10阐明了基于是否已知未压缩大小，流式LZMA文件和非流式LZMA文件之间的区别。

### **3.4 压缩数据 (剩余部分)**

这是经过LZMA算法压缩后的原始比特流，它是一系列使用范围编码技术编码的比特。

相比之下，.xz文件格式是一种更新的容器格式，旨在取代传统的.lzma格式 [25]。它以6字节的魔数 "FD 37 7A 58 5A 00" 开头 26，支持链式使用多种压缩算法（过滤器），其中LZMA2是主要的压缩算法 [2]。.xz 格式还包含完整性校验（如CRC32、CRC64、SHA256）[2] 和对多个独立压缩块（block）的支持，这些块通过一个尾部索引进行管理 [26]。Snippet [26] 强调了 .xz 相比 .lzma 更复杂的结构，这使得诸如多块文件中的随机访问解压缩等功能成为可能。

LZMA2格式是一种基于LZMA的简单容器格式，它提供了更好的多线程支持，并且能够高效地压缩部分不可压缩的数据，因为它允许存储未压缩的数据块 [6]。LZMA2的头部包含一个字节，用于指示字典大小（使用特定的LZMA2方案编码，支持有限的尺寸集合），以及一个字节用于存储LZMA模型属性（lc、lp、pb） [14]。LZMA2还规定了字面值上下文位数（lc）和字面值位置位数（lp）之和不得超过4 [15]。

**.lzma 文件格式总结表：**

| 偏移 (字节) | 大小 (字节) | 描述 |
| :---- | :---- | :---- |
| 0 | 1 | LZMA模型属性（以编码形式表示的lc、lp、pb） |
| 1 | 4 | 字典大小（32位无符号整数，小端序） |
| 5 | 8 | 未压缩大小（64位无符号整数，小端序） |
| 13 | 可变 | 压缩数据（LZMA流） |

## **4. LZMA中的马尔科夫链概率预测**

LZMA算法在预测下一个比特或符号的概率时，采用了基于马尔科夫链的上下文模型。

### **4.1 马尔科夫链简介**

马尔科夫链是一种随机模型，其未来状态的概率仅取决于当前状态，而与之前的状态无关（即具有无记忆性） [31]。在 LZMA 中，马尔科夫链被用来构建一个上下文模型，从而根据先前数据的上下文信息来预测下一个比特或符号（字面值或匹配的一部分）的概率 [14]。Snippet [46] 提到，马尔科夫链被应用于现代压缩器中，例如 7-Zip（它使用LZMA）。

### **4.2 上下文建模**

LZMA 采用了一种针对字面值和匹配的比特字段的特定上下文建模方法，这与通用的基于字节的模型不同 [14]。这种特定的建模方式避免了在同一上下文中混合不相关的比特，从而提高了压缩率 [14]。预测下一个比特的上下文是由先前的比特或字节的值决定的 [14]。

### **4.3 上下文参数**

在 LZMA 中，"字面值上下文位数" (lc)、"字面值位置位数" (lp) 和 "位置位数" (pb) 这三个参数在定义上下文时起着关键作用 [3]：

* **lc (字面值上下文位数):** 指的是前一个未压缩字节（字面值）的高位比特数，用于形成预测当前字面值比特的上下文。较高的 lc 值（最大为 8）使得模型能够考虑更多前一个字面值的信息 [3]。这实际上为字面值创建了一个更大的马尔科夫链状态空间。
* **lp (字面值位置位数):** 指的是当前未压缩数据位置的低位比特数，用于进一步细化字面值预测的上下文。这有助于捕获可能依赖于字面值在特定对齐位置（模2lp）的模式 [3]。对于在特定字节偏移处具有重复结构的数据非常有用。
* **pb (位置位数):** 指的是当前未压缩数据位置的低位比特数，用于形成预测编码类型（字面值、匹配、重复）的上下文。这有助于模型根据数据流中的一般位置（模2 pb）调整其预测 [3]。

### **4.4 状态转移和概率模型**

LZMA 算法采用一个包含12个状态的状态机来跟踪最近编码的数据包（字面值、匹配、长重复、短重复）的历史 [14]。初始状态为 0，假设之前的包都是字面值 [14]。这些状态会影响用于编码下一个数据包的概率模型。状态之间的转移是确定性的，取决于刚刚编码的数据包的类型 14。对于要编码的每个比特，LZMA 使用一个上下文（由状态和 lc、lp、pb 决定）来选择一个特定的概率估计器。这个估计器预测该比特为0的概率。Snippet [14] 提到，这个概率通常存储为一个 11 位的无符号整数。

### **4.5 自适应概率更新**

LZMA使用的概率估计器是自适应的。在每个比特被编码或解码之后，相应的概率估计会被更新，以更好地反映在该特定上下文中观察到的0和1的频率 14。LZMA对这些概率模型使用了一种分数更新方法 [36]。Snippet [18] 提供了具体的更新规则：如果解码的符号是 0，则增加 0 的概率；如果是 1，则增加 1 的概率。增加/减少的幅度由一个移位因子（kNumMoveBits，通常为 5）控制。这种持续的适应性使得编码器和解码器能够为每个上下文维护同步的概率估计，这对于高效的范围编码至关重要，并最终带来更高的压缩率 [22]。

LZMA中基于马尔科夫链的概率预测是一项关键技术，它通过使用由lc、lp和pb等参数以及状态机跟踪的近期编码事件历史精心构建的上下文，实现了卓越的压缩性能。概率模型的自适应特性确保了模型能够学习并适应被压缩数据的特定特征。

## **5. 范围编码的实现细节**

范围编码是 LZMA 算法采用的熵编码技术，用于高效地表示经过 LZ77 阶段处理后的匹配和字面值信息。

### **5.1 范围编码简介**

LZMA 使用范围编码取代了更常见的霍夫曼编码 [1]。范围编码通过将一系列符号表示为 0 到 1 范围内的单个分数来实现压缩。符号出现的概率越高，其在范围中所占的区间就越小，从而实现高效编码 [22]。范围编码相比霍夫曼编码提供了更高的压缩效率，因为它能够更接近数据的理论熵极限，尤其是在处理概率不是 1/2 的幂的符号时 [1]。在 LZMA 中，范围编码器操作的是一个二进制字母表（0和1），因此范围的划分是一个二元过程 [22]。

### **5.2 范围编码原理**

编码器维护一个当前的范围。对于要编码的每个符号，这个范围被划分为多个子范围，每个子范围的大小与对应符号的概率成正比 [1]。与要编码的符号对应的子范围成为新的当前范围。解码器从相同的初始范围开始，并通过检查编码的比特流，确定编码器在每个步骤中选择了哪个子范围，从而重构原始的符号序列 [1]。

### **5.3 LZMA范围编码器和解码器的实现**

LZMA 解码器维护两个关键的 32 位无符号整数变量：Range（当前范围）和Code（来自编码比特流的位于当前范围内的值） [14]。解码器通过将 Range 设置为0xFFFFFFFF并读取压缩流的前5个字节来初始化（第一个字节被忽略，接下来的四个字节构成初始代码） 14。要使用给定的概率（prob）解码一个比特，解码器计算一个bound \= (Range \>\> kNumBitModelTotalBits) \* prob。如果 Code \< bound，则解码的比特为 0，新的 Range 变为 bound。否则，解码的比特为1，Code减去bound，新的 Range 变为 Range \- bound 14。编码器使用 low 和 width 来定义当前范围。对于每个比特，它会根据概率计算一个阈值。然后根据要编码的比特更新范围 [37]。

### **5.4 归一化过程**

在编码和解码过程中，随着范围的缩小，可能会丢失精度。为了防止这种情况，需要执行归一化步骤：如果在解码器中 Range 或在编码器中 width 降到某个阈值以下（例如，解码器中为 224 [18]），则范围和代码（或 low 和 width）会左移 8 位，并且从输入流中读取下一个字节到 Code 的低 8 位（对于解码器），或者输出一个字节（对于编码器）[14]。Snippet [47] 讨论了在此过程中如何处理进位。

### **5.5 概率处理**

LZMA 中的范围编码过程严重依赖于马尔科夫链上下文模型提供的概率。比特为 0 的概率（由11位的 prob 值表示）决定了在编码和解码过程中如何划分当前范围 14。

### **5.6 与霍夫曼编码的比较**

范围编码相比霍夫曼编码能够实现更高的压缩率，尤其是在处理概率不为2的幂或者符号概率不是整数比特长度的情况下 1。与为符号分配整数比特长度的霍夫曼编码不同，范围编码可以使用分数比特，从而更有效地表示接近其熵的数据 1。LZMA使用范围编码避免了对固定码本的需求，因为概率是基于上下文模型动态处理的 [1]。

范围编码是一种复杂但功能强大的熵编码技术，它通过将数据流表示为一个非常精确的分数，使得LZMA能够实现高压缩效率。与自适应马尔科夫链概率模型的紧密集成是其有效性的关键。归一化过程确保了编码和解码的精度不会随着范围的缩小而降低。使用来自马尔科夫链模型的概率使得范围编码器能够高度适应输入数据。

## **6. LZMA关键部分的C++伪代码实现**

### **6.1 滑动窗口内的匹配算法**

C++ 伪代码：

```cpp
// 假设 'data' 是输入缓冲区，'window' 是滑动窗口，
// 'windowSize' 是窗口的当前大小，'lookaheadSize' 是前瞻缓冲区的大小。

function findLongestMatch(data, currentPosition, window, windowSize, lookaheadSize):
    bestMatchLength = 0
    bestMatchDistance = 0

    for distance from 1 to windowSize:
        maxLength = 0
        while currentPosition + maxLength < length(data) and
              windowSize - distance + maxLength >= 0 and
              data[currentPosition + maxLength] == window:
            maxLength = maxLength + 1

        if maxLength > bestMatchLength and maxLength >= MIN_MATCH_LENGTH: // MIN_MATCH_LENGTH 通常为 2 或 3
            bestMatchLength = maxLength
            bestMatchDistance = distance

    return (bestMatchDistance, bestMatchLength)
```

**注释:** 这段伪代码展示了在滑动窗口中查找最长匹配序列的基本方法。它遍历可能的距离，并将前瞻缓冲区中的数据与窗口中相应位置的数据进行比较。

### **6.2 使用马尔科夫链上下文的概率预测**

C++ 伪代码：

```cpp
// 假设 'probabilityTable' 是概率值数组 (0-2047)，
// 'context' 是基于 lc、lp、pb 和先前数据计算得到的索引。

function getProbability(probabilityTable, context):
    return probabilityTable[context]

function updateProbability(probabilityTable, context, bit):
    probability = probabilityTable[context]
    if bit == 0:
        probability = probability + ((1 << 11) - probability) >> 5 // 示例更新规则 (kNumMoveBits = 5)
    else:
        probability = probability - (probability >> 5)
    probabilityTable[context] \= probability
```

**注释:** 这段伪代码展示了如何使用上下文从概率表中获取概率。updateProbability 函数展示了一个基于解码比特的简化的概率更新机制。

### **6.3 范围编码和解码过程**

C++ 伪代码：

```cpp
// --- 范围编码 ---
function encodeBit(bit, probability):
    threshold = width * probability
    if bit == 0:
        width = threshold
    else:
        low = low + threshold
        width = width - threshold
    while width < NORMALIZATION_THRESHOLD:
        width = width << 8
        low = low << 8 + nextBitFromInput // 为了演示，实际输出更复杂

// --- 范围解码 ---
function decodeBit(probability):
    threshold = range * probability
    if code < threshold:
        range = threshold
        return 0
    else:
        code = code - threshold
        range = range - threshold
        return 1
    while range < NORMALIZATION_THRESHOLD:
        range = range << 8
        code = code << 8 + nextByteFromStream
```

**注释:** 这些伪代码片段提供了基于给定概率对单个比特进行范围编码和解码的核心逻辑的高级概述。还包括归一化步骤。

## **7. LZMA 实现中的代码优化核心原理（基于xz  utils）**

XZ Utils 作为一个生产级的实现，非常注重性能。通过分析其 `src/liblzma/` 目录 [28] 及其子目录（如 `lzma/` [38]），可以发现许多针对特定架构的优化。

XZ Utils 在 x86 平台上的解码器中使用了 SIMD（单指令多数据流）指令，如 SSE2 [39]，用于并行处理数据，这显著加快了 LZMA 解码器中内存复制和比较等操作的速度。它还针对循环冗余校验（CRC）实现了高度优化的版本，利用硬件指令（例如，x86-64上的 CLMUL [39] 和 LoongArch 等架构上的专用 CRC32 指令 [39]）来快速进行数据完整性验证。此外，XZ Utils 还包含了针对特定处理器架构的代码路径，以利用其独特的特性，例如，在支持快速非对齐访问的 64 位 PowerPC 和 RISC-V 处理器上提高了 LZMA/LZMA2 编码器的速度 [39]。

XZ Utils 很可能在 LZMA 实现的各个部分使用了查找表，例如，在匹配查找算法（如哈希链）中进行快速哈希，以及在范围编码器中进行更快的概率到范围计算。（需要进一步分析源代码以获取具体示例）。由于LZMA在比特级别上运行，XZ Utils 代码库会大量使用位运算符（AND、OR、XOR、移位）来快速操作标志、概率和编码数据。仔细组织数据结构以方便高效的比特访问也是一种常见的优化方法。（需要分析源代码）。诸如 XZ Utils 之类的实现旨在最小化内存访问，特别是对于像滑动窗口和概率数组这样的常用数据结构，以提高缓存命中率。使用适当的小数据类型和在内存中连续组织数据是常见的技术。（需要分析源代码）。

LZMA2 是 .xz 格式的主要压缩算法，与 LZMA 相比，它在设计上就具有更好的多线程支持 [6]。XZ Utils 利用了这一点，提供了多线程压缩功能，允许它利用多个CPU核心来加速大型文件的压缩过程 [2]。Snippet 13表明可以通过命令行参数控制多线程。

XZ Utils项目通过集成特定于架构的优化、高效的算法和并行处理能力，展现了对性能的高度关注。这些优化对于使LZMA和LZMA2能够实际应用于涉及大量数据的现实世界场景至关重要。检查源代码将提供这些技术的具体示例。实际的压缩工具必须经过高度优化才能可用。通过研究广泛使用且备受推崇的实现（如XZ Utils），我们可以了解在LZMA中实现高性能所采用的实用技术。SIMD指令、硬件CRC和特定于架构的代码的使用表明了对处理器能力的深入理解。提高缓存利用率和减少内存访问的策略是性能的基础。LZMA2中的多线程支持及其在XZ Utils中的实现突出了利用多核架构来加快压缩速度的重要性。

## **8. 结论**

LZMA 算法的核心原理包括：基于滑动窗口的字典压缩（源于 LZ77）、使用自适应马尔科夫链上下文模型进行概率预测以及使用范围编码进行高效的熵编码。LZMA 的优势在于其极高的压缩率、相对较快的解压缩速度以及通过各种参数（lc、lp、pb、字典大小、匹配查找器）实现的高度适应性。然而，与其他一些算法（如 gzip，尤其是在较高压缩级别下）相比，其压缩速度可能较慢，并且可能具有更高的内存使用量，尤其是在使用较大字典大小进行压缩时。LZMA 广泛应用于创建高压缩率的归档文件（例如，.7z、.tar.xz）、软件分发（包括 Debian、Fedora、Arch Linux 中的操作系统软件包 [2]）、嵌入式系统中的固件和资源压缩 4 以及各种领域中大型数据集的压缩。Snippet [9] 概述了其应用领域。无损压缩的未来趋势和潜在改进包括：进一步提高压缩率，提高压缩和解压缩速度（可能通过更先进的并行化技术 2），减少内存占用，以及开发对数据损坏更具弹性的算法 [1]。诸如 Zstandard [2] 等新压缩算法的出现表明该领域正在不断发展。

## **引用的著作**

（访问时间为 2025年4月10日）

1. LZMA File Format \- Lempel Ziv Markov Chain Algorithm \- Aspose, [https://products.aspose.com/zip/most-common-archives/what-is-lzma/](https://products.aspose.com/zip/most-common-archives/what-is-lzma/)
2. XZ Utils \- Wikipedia, [https://en.wikipedia.org/wiki/XZ\_Utils](https://en.wikipedia.org/wiki/XZ_Utils)
3. 7zip/DOC/lzma.txt at master · yumeyao/7zip · GitHub, [https://github.com/yumeyao/7zip/blob/master/DOC/lzma.txt](https://github.com/yumeyao/7zip/blob/master/DOC/lzma.txt)
4. LZMA SDK (Software Development Kit) \- 7-Zip, [https://www.7-zip.org/sdk.html](https://www.7-zip.org/sdk.html)
5. upx-lzma-sdk/lzma.txt at master \- GitHub, [https://github.com/upx/upx-lzma-sdk/blob/master/lzma.txt](https://github.com/upx/upx-lzma-sdk/blob/master/lzma.txt)
6. 7z Format \- 7-Zip 18 Documentation, [https://documentation.help/7-Zip-18.0/7z.htm](https://documentation.help/7-Zip-18.0/7z.htm)
7. 7z Format \- 7-Zip Documentation, [https://documentation.help/7-Zip/7z.htm](https://documentation.help/7-Zip/7z.htm)
8. 7z Format \- 7-Zip, [https://www.7-zip.org/7z.html](https://www.7-zip.org/7z.html)
9. How to Install and Use LZMA Compression on Linux, [https://pendrivelinux.com/lzma-compression/](https://pendrivelinux.com/lzma-compression/)
10. lzma, [http://man.he.net/man1/lzma](http://man.he.net/man1/lzma)
11. lzma.txt \- MIT, [http://web.mit.edu/outland/arch/i386\_rhel4/build/p7zip-current/DOCS/lzma.txt](http://web.mit.edu/outland/arch/i386_rhel4/build/p7zip-current/DOCS/lzma.txt)
12. emCompress-LZMA User Guide & Reference Manual \- SEGGER Online Documentation, [https://doc.segger.com/UM17002\_emCompress\_LZMA.html](https://doc.segger.com/UM17002_emCompress_LZMA.html)
13. \-m (Set compression Method) switch \- 7-Zip, [https://7-zip.opensource.jp/chm/cmdline/switches/method.htm](https://7-zip.opensource.jp/chm/cmdline/switches/method.htm)
14. Lempel–Ziv–Markov chain algorithm \- Wikipedia, [https://en.wikipedia.org/wiki/Lempel%E2%80%93Ziv%E2%80%93Markov\_chain\_algorithm](https://en.wikipedia.org/wiki/Lempel%E2%80%93Ziv%E2%80%93Markov_chain_algorithm)
15. LZMA compression settings details \- Stack Overflow, [https://stackoverflow.com/questions/3057171/lzma-compression-settings-details](https://stackoverflow.com/questions/3057171/lzma-compression-settings-details)
16. xz-utils/NEWS at master\_jammy \- GitHub, [https://github.com/pop-os/xz-utils/blob/master\_jammy/NEWS](https://github.com/pop-os/xz-utils/blob/master_jammy/NEWS)
17. 7-Zip Manual lzma.txt, [https://7zip.bugaco.com/7zip/lzma.txt](https://7zip.bugaco.com/7zip/lzma.txt)
18. LZMA-SDK/DOC/lzma-specification.txt at master · jljusten/LZMA ..., [https://github.com/jljusten/LZMA-SDK/blob/master/DOC/lzma-specification.txt](https://github.com/jljusten/LZMA-SDK/blob/master/DOC/lzma-specification.txt)
19. Hardware Implementation of LZMA Data Compression Algorithm \- International Journal of Applied Information Systems, [https://research.ijais.org/volume5/number4/ijais12-450900.pdf](https://research.ijais.org/volume5/number4/ijais12-450900.pdf)
20. Discovering Dataset Nature through Algorithmic Clustering based on String CompressionThis is the postprint version of an article published in IEEE TKDE. The final published version is available at https://doi.org/10.1109/TKDE.2014.2345396. ©2015 IEEE. \- arXiv, [https://arxiv.org/html/2502.00208v1](https://arxiv.org/html/2502.00208v1)
21. Hardware Implementation of LZMA Data Compression Algorithm \- ResearchGate, [https://www.researchgate.net/publication/275038202\_Hardware\_Implementation\_of\_LZMA\_Data\_Compression\_Algorithm](https://www.researchgate.net/publication/275038202_Hardware_Implementation_of_LZMA_Data_Compression_Algorithm)
22. LZMA compression explained \- Gautier's blog, [https://gautiersblog.blogspot.com/2016/08/lzma-compression.html](https://gautiersblog.blogspot.com/2016/08/lzma-compression.html)
23. Lempel-Ziv-Markov Chain Algorithm Modeling using Models of Computation and ForSyDe, [https://www.researchgate.net/publication/336795137\_Lempel-Ziv-Markov\_Chain\_Algorithm\_Modeling\_using\_Models\_of\_Computation\_and\_ForSyDe](https://www.researchgate.net/publication/336795137_Lempel-Ziv-Markov_Chain_Algorithm_Modeling_using_Models_of_Computation_and_ForSyDe)
24. A very brief BitKnit retrospective \- The ryg blog \- WordPress.com, [https://fgiesen.wordpress.com/2023/05/06/a-very-brief-bitknit-retrospective/](https://fgiesen.wordpress.com/2023/05/06/a-very-brief-bitknit-retrospective/)
25. xz/doc/lzma-file-format.txt at master · tukaani-project/xz · GitHub, [https://github.com/tukaani-project/xz/blob/master/doc/lzma-file-format.txt](https://github.com/tukaani-project/xz/blob/master/doc/lzma-file-format.txt)
26. XZ/LZMA Worked Example Part 5: XZ \- Nigel Tao, [https://nigeltao.github.io/blog/2024/xz-lzma-part-5-xz.html](https://nigeltao.github.io/blog/2024/xz-lzma-part-5-xz.html)
27. lzma-sys 0.1.8 \- Docs.rs, [https://docs.rs/crate/lzma-sys/0.1.8/source/xz-5.2.3/doc/xz-file-format.txt](https://docs.rs/crate/lzma-sys/0.1.8/source/xz-5.2.3/doc/xz-file-format.txt)
28. tukaani-project/xz: XZ Utils \- GitHub, [https://github.com/tukaani-project/xz](https://github.com/tukaani-project/xz)
29. Debian \-- Details of package xz-utils in sid, [https://packages.debian.org/sid/xz-utils](https://packages.debian.org/sid/xz-utils)
30. lzma — Compression using the LZMA algorithm — Python 3.13.3 documentation, [https://docs.python.org/3/library/lzma.html](https://docs.python.org/3/library/lzma.html)
31. Markov Chain Explained | Built In, [https://builtin.com/machine-learning/markov-chain](https://builtin.com/machine-learning/markov-chain)
32. Markov chain \- Wikipedia, [https://en.wikipedia.org/wiki/Markov\_chain](https://en.wikipedia.org/wiki/Markov_chain)
33. Chapter 8: Markov Chains, [https://www.stat.auckland.ac.nz/\~fewster/325/notes/ch8.pdf](https://www.stat.auckland.ac.nz/~fewster/325/notes/ch8.pdf)
34. How to Predict Sales Using Markov Chain, [https://blog.arkieva.com/markov-chain-sales-prediction/](https://blog.arkieva.com/markov-chain-sales-prediction/)
35. LZMA parametrization \- Gautier's blog, [https://gautiersblog.blogspot.com/2016/09/lzma-parametrization.html](https://gautiersblog.blogspot.com/2016/09/lzma-parametrization.html)
36. 06-12-14 \- Some LZMA Notes \- cbloom rants, [http://cbloomrants.blogspot.com/2014/06/06-12-14-some-lzma-notes.html](http://cbloomrants.blogspot.com/2014/06/06-12-14-some-lzma-notes.html)
37. XZ/LZMA Worked Example Part 1: Range Coding \- Nigel Tao, [https://nigeltao.github.io/blog/2024/xz-lzma-part-1-range-coding.html](https://nigeltao.github.io/blog/2024/xz-lzma-part-1-range-coding.html)
38. liblzma (XZ Utils): lzma Directory Reference, [https://tukaani.org/xz/liblzma-api/dir\_b17a1d403082bd69a703ed987cf158fb.html](https://tukaani.org/xz/liblzma-api/dir_b17a1d403082bd69a703ed987cf158fb.html)
39. Releases · tukaani-project/xz \- GitHub, [https://github.com/tukaani-project/xz/releases](https://github.com/tukaani-project/xz/releases)
40. XZ Utils 5.8 Introduces Performance Improvements in the LZMA/LZMA2 Decoder \- 9to5Linux, [https://9to5linux.com/xz-utils-5-8-introduces-performance-improvements-in-the-lzma-lzma2-decoder](https://9to5linux.com/xz-utils-5-8-introduces-performance-improvements-in-the-lzma-lzma2-decoder)
41. xz/NEWS at master · tukaani-project/xz \- GitHub, [https://github.com/tukaani-project/xz/blob/master/NEWS](https://github.com/tukaani-project/xz/blob/master/NEWS)
42. XZ Utils Backdoor: Supply Chain Vulnerability (CVE-2024-3094) \- Logpoint, [https://www.logpoint.com/en/blog/emerging-threats/xz-utils-backdoor/](https://www.logpoint.com/en/blog/emerging-threats/xz-utils-backdoor/)
43. Thoughts on the xz backdoor: an lzma-rs perspective | Blog \- Guillaume Endignoux, [https://gendignoux.com/blog/2024/04/08/xz-backdoor.html](https://gendignoux.com/blog/2024/04/08/xz-backdoor.html)
44. xz/TODO at master · tukaani-project/xz \- GitHub, [https://github.com/tukaani-project/xz/blob/master/TODO](https://github.com/tukaani-project/xz/blob/master/TODO)
45. Increase robustness of 7z LZMA archiving? \- backup \- Super User, [https://superuser.com/questions/1756414/increase-robustness-of-7z-lzma-archiving](https://superuser.com/questions/1756414/increase-robustness-of-7z-lzma-archiving)
46. Markov Chain Compression (Ep 3, Compressor Head) \- YouTube, [https://www.youtube.com/watch?v=05RFEGWNxts](https://www.youtube.com/watch?v=05RFEGWNxts)
47. Somewhere Range Coding went off the rails \- Richard Geldreich's Blog, [http://richg42.blogspot.com/2023/04/somewhere-range-coding-went-off-rails.html](http://richg42.blogspot.com/2023/04/somewhere-range-coding-went-off-rails.html)
