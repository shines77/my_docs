# **动态马尔科夫压缩算法原理与C++实现**

## 概述

* 来自：[Gemini 2.5 Pro](https://gemini.google.com/)
* 网址：[Gemini Deep Research](https://gemini.google.com/app/9cf341902f338c4e?hl=zh-cn)
* Google Docs：[Google 文档](https://docs.google.com/document/d/1l3GOImX_kPy79eA_gwx3CV7IbtuHJiBabTUxbihg4G0/edit?tab=t.0)

## 1. 动态马尔科夫压缩算法概述 (Overview of Dynamic Markov Compression):

   无损数据压缩在现代信息技术中扮演着至关重要的角色，它能够在不损失任何原始数据信息的前提下，显著减小数据的体积，从而提高存储效率和传输速度。在众多无损压缩算法中，例如 Huffman 编码 1 和 LZW 算法 2，它们各有优势和局限性。为了进一步提高压缩率，尤其是在处理具有复杂统计特性的数据时，研究人员不断探索新的算法。动态马尔科夫压缩算法 (Dynamic Markov Compression, DMC) 正是在这样的背景下由 Gordon Cormack 和 Nigel Horspool 开发出来的一种高效的无损数据压缩方法 4。该算法最初在他们 1987 年发表的论文 "Data Compression Using Dynamic Markov Modelling" 中被提出 3。

   DMC 的核心思想是利用动态构建的马尔科夫模型来预测输入数据流中的每一个比特，并结合算术编码对预测结果进行高效编码 4。这种方法的核心在于根据先前已经处理过的比特序列（称为上下文）来预测下一个比特是 0 还是 1 的概率。马尔科夫模型能够捕获数据中相邻比特之间的依赖关系，并随着数据的处理不断学习和调整这些关系 6。

   DMC 的一个关键特点是它逐比特地进行预测和编码 4，这与逐字节操作的预测部分匹配 (Prediction by Partial Matching, PPM) 算法形成了鲜明的对比 4。逐比特的方法使得 DMC 能够捕捉到数据中更为细微的统计依赖性，从而在理论上可以实现更高的压缩率，尤其是在处理那些比特之间存在复杂关联或者数据不自然地按字节对齐的情况下。

## 2. 动态马尔科夫压缩算法的原理 (Principles of Dynamic Markov Compression):

   在深入了解 DMC 的工作原理之前，回顾一下马尔科夫模型的基础概念是很有必要的。马尔科夫模型是一种随机过程，其未来状态的概率分布仅依赖于当前状态，而与系统之前的历史状态无关 7。这种“无记忆性”是马尔科夫模型的关键特征 7。一个马尔科夫模型由一组可能的状态、状态之间的转移以及与这些转移相关的概率组成 6。马尔科夫模型在数据压缩领域的应用主要是通过构建一个能够预测下一个符号（在 DMC 中，这个符号是比特）概率的模型，然后利用这些概率进行高效的编码 2。

   DMC 的核心思想在于进行比特级的预测并建立相应的上下文模型 4。与传统的基于字节的压缩算法不同，DMC 关注的是输入数据流中的每一个比特，并尝试根据先前已经处理过的比特序列（即上下文）来预测当前比特是 0 还是 1 的概率 4。为了实现这一目标，DMC 使用上下文模型，该模型通过跟踪在每个特定的上下文中 0 和 1 的出现频率来估计下一个比特的概率 4。

   DMC 的工作流程主要包括以下几个步骤：

   * **初始化模型：** DMC 通常从一个包含一组预定义的短上下文开始构建其模型，这些上下文的长度可能在 8 到 15 位之间，并且通常与字节边界对齐 4。初始状态可以是这些预定义上下文中的任何一个，例如一个 8 位的上下文 4。为了避免在首次遇到某个上下文时预测概率为零的情况，模型中 0 和 1 的计数器通常会被初始化为一个小的非零常数，例如 0.2 或者 1 4。

   * **比特预测：** 对于每一个需要压缩的比特，DMC 的预测器会基于当前的上下文来预测下一个比特是 0 还是 1 的概率。这个概率的计算通常基于当前上下文中已经观察到的 0 和 1 的计数。具体来说，如果在一个给定的上下文中，比特 0 已经出现了 *n*\0\ 次，比特 1 已经出现了 *n*\1\ 次，那么预测下一个比特为 0 的概率就是 *n*\0\ / (*n*\0\ \+ *n*\1\)，而预测为 1 的概率则是 *n*\1\ / (*n*\0\ \+ *n*\1\) 4。

   * **算术编码：** 一旦获得了下一个比特的预测概率，DMC 就使用算术编码器来对实际出现的比特进行编码 4。算术编码是一种高效的熵编码方法，它通过将整个消息表示为 0 到 1 之间的一个实数来工作。编码器根据每个符号（在 DMC 中是比特）的预测概率，逐步缩小代表整个消息的概率区间。最终，压缩后的代码就是能够唯一标识原始消息的这个实数的一个二进制表示。DMC 通常使用比特级的算术编码器 4。

   * **模型更新：** 在编码或解码完一个比特后，DMC 会根据实际观察到的比特来更新其马尔科夫模型 4。这通常涉及到在当前的上下文中，将对应于实际出现的比特的计数器递增。模型还需要确定在处理完当前比特后应该转换到的下一个上下文。这通常通过在当前上下文中维护指向下一个可能上下文的链接（指针）来实现，每个上下文都可能包含两个链接，一个对应于下一个比特是 0 的情况，另一个对应于下一个比特是 1 的情况 4。

   * **上下文动态扩展（克隆）：** 为了进一步提高压缩率，DMC 具备动态扩展上下文的能力，即在压缩过程中创建新的、更长的上下文 4。当算法从一个上下文 A 转换到另一个上下文 B 时（上下文 B 通常是通过从 A 的左侧丢弃若干比特得到的），如果从 A 到 B 的转换非常频繁，并且上下文 B 本身也经常被访问，DMC 可能会创建一个新的上下文 C。上下文 C 表示与上下文 A 附加当前比特相同的比特序列，但与 B 不同的是，C 不会丢弃左侧的任何比特 4。然后，上下文 A 中指向 B 的链接会被更新为指向新创建的上下文 C。最初，上下文 B 和 C 会做出相同的预测，并且它们会指向相同的后续状态 4。上下文 C 的总计数会被设置为等于上下文 A 中与当前输入比特对应的计数，并且这个计数会从上下文 B 的计数中减去 4。这种动态创建更长上下文的机制使得 DMC 能够更好地捕获数据中可能存在的长距离依赖关系。

与 DMC 类似的算法还包括预测部分匹配 (PPM)。DMC 和 PPM 都使用预测算术编码，并且都是基于上下文的统计压缩方法 4。然而，它们之间最主要的区别在于 DMC 逐比特地进行编码，而 PPM 通常是逐字节地进行编码 4。这种逐比特的操作使得 DMC 在理论上可以获得更高的压缩率，因为它能够捕捉到更细微的比特级别的依赖关系。此外，DMC 与像 PAQ 这样的上下文混合算法也存在不同。在 PAQ 等算法中，对于每一次预测，它们会将来自多个不同阶数的上下文模型的预测进行混合，以获得更准确的概率估计 4。相比之下，DMC 在每次预测时通常只依赖于一个当前的上下文 4。总的来说，DMC 的压缩率和速度通常与 PPM 相当，但 DMC 需要相对更多的内存来存储其动态构建的马尔科夫模型，并且由于其比特级的操作，可能在速度上略逊于某些优化的 PPM 实现。此外，DMC 的实现相对复杂，因此不如 PPM 广泛使用 4。

## 3. 动态马尔科夫压缩算法在无损数据压缩领域的应用和优势 (Applications and Advantages of Dynamic Markov Compression in Lossless Data Compression):

   动态马尔科夫压缩算法作为一种通用的无损压缩算法，理论上可以应用于各种类型的数据，包括文本、图像、音频和可执行文件等 15。早期的研究和实验结果表明，DMC 在某些类型的数据上，特别是那些具有复杂比特级依赖关系的数据上，可以获得比其他传统的无损压缩算法（如自适应 Huffman 编码和 Ziv-Lempel 算法）更好的压缩率 1。例如，在对象代码文件等非均匀数据上，DMC 表现出了其独特的优势 2。
   尽管 DMC 在通用数据压缩领域展现了潜力，但其在特定领域的应用也值得关注。例如，由于其基于比特的处理能力，DMC 可能特别适用于对序列数据（如基因组数据）进行高效压缩，因为基因组数据在比特层面可能存在复杂的统计模式 12。事实上，一些研究项目已经尝试将 DMC 应用于 DNA 压缩，并取得了一定的成果，例如 rajatdiptabiswas 的 DNA 压缩项目就包含了一个 C 语言的 DMC 实现 12。

   相较于传统的无损压缩算法，DMC 主要具有以下优势：

   * **高压缩率：** 实验结果通常表明，DMC 在许多类型的数据上能够实现比 Huffman 和 LZW 等算法更高的压缩率，尤其是在那些具有复杂统计特性的数据上 1。这主要是因为 DMC 逐比特地进行建模和预测，能够更精细地捕捉数据中的冗余信息。

   * **自适应性：** DMC 模型在压缩过程中是动态构建和调整的，它不需要事先对数据进行详细的统计分析或构建静态模型 1。这种自适应性使得 DMC 能够有效地处理不同来源和特性的数据，即使数据的统计特性在压缩过程中发生变化，DMC 也能够进行相应的调整。

   * **对不同类型数据的适应性：** 由于其基于比特的预测机制，DMC 不像那些基于特定符号或模式匹配的算法那样对特定类型的数据有很强的偏好，理论上可以有效地压缩各种二进制数据，包括那些结构不规则或已经进行过初步压缩的数据 2。

   然而，DMC 也存在一些局限性和挑战，这可能解释了为什么它没有像其他一些无损压缩算法那样得到广泛的应用：

   * **计算复杂度：** 逐比特的处理以及动态维护和扩展马尔科夫模型可能导致较高的计算成本，使得压缩和解压缩的速度相对较慢，这在对实时性要求高的应用中是一个明显的缺点 4。

   * **内存占用：** 为了实现较高的压缩率，DMC 需要维护一个不断增长的马尔科夫模型，这可能导致较高的内存需求，尤其是在处理大型文件时 4。模型中状态的数量会随着输入数据的复杂性和长度而增加。

   * **实现难度：** DMC 的原理相对复杂，特别是上下文动态扩展（克隆）和与算术编码器的集成，使得其实现比一些更简单的压缩算法更具挑战性 6。确保编码器和解码器在模型演化上保持同步也增加了实现的难度。

## 4. 动态马尔科夫压缩算法的 C++ 代码实现 (C++ Code Implementation of Dynamic Markov Compression):

   实现动态马尔科夫压缩算法的 C++ 代码涉及到几个关键的数据结构和步骤。

   在核心数据结构设计方面，首先需要定义用于表示马尔科夫模型状态的结构。每个状态都应该能够存储在该特定上下文中观察到的 0 和 1 的计数，这些计数将用于计算预测下一个比特的概率 4。此外，每个状态还需要包含指向下一个状态的指针或索引，分别对应于输入比特为 0 和 1 的情况，以实现状态之间的转移 4。为了支持上下文的动态扩展（克隆）机制，状态结构可能还需要存储指向更长上下文的链接或者包含创建新状态所需的信息 4。
   其次，需要一种数据结构来存储状态转移的计数。可以使用二维数组或者哈希表来实现，其中一个维度表示当前状态，另一个维度表示可能的输入比特（0 或 1），存储的值则是在该状态下接收到该输入比特后发生的转移次数 13。选择哪种数据结构取决于对查找速度和内存效率的需求。

   最后，算术编码器的实现是 DMC 的另一个关键组成部分。这需要实现算术编码的核心逻辑，包括维护表示当前概率范围的低位和高位数值，并根据模型的预测概率对这些范围进行划分和更新 4。同时，为了能够恢复原始数据，还需要实现相应的算术解码逻辑。

   在编码器的 C++ 实现中，第一步通常是初始化模型，即创建初始状态并初始化相关的计数器和指针 4。例如，可以创建一个根状态，其 0 和 1 的计数都初始化为某个小的非零值。接下来，需要实现比特预测函数，该函数接收当前状态作为输入，并根据该状态中存储的 0 和 1 的计数来计算下一个比特为 0 或 1 的概率 4。然后，算术编码函数会接收要编码的实际比特和预测概率作为输入，更新算术编码器的内部状态，并输出相应的压缩比特流 4。模型更新函数则根据实际编码的比特来更新当前状态的计数器，并根据状态转移规则移动到下一个状态 4。最后，上下文克隆机制的实现需要在满足预定义的克隆条件时，动态地创建新的状态，并更新状态之间的连接关系，以实现更精细的上下文建模 4。

   解码器的 C++ 实现需要与编码器保持同步。首先，解码器需要使用与编码器完全相同的初始模型 4。然后，它需要实现与编码器相同的比特预测函数 4。算术解码函数会读取压缩的比特流，并使用当前的预测概率来确定原始的比特值，同时更新算术解码器的内部状态 4。模型更新函数与编码器类似，根据解码出的比特来更新其维护的马尔科夫模型，并移动到下一个状态 4。最后，解码器也需要实现与编码器相同的上下文克隆逻辑，以确保在解码过程中构建出与编码器相同的模型 4。

   虽然研究材料中提到了一些 DMC 的 C 实现 (6) 和相关的算术编码实现 (20)，但并没有直接提供完整的 DMC C++ 代码示例。报告在后续部分将讨论一些已知的 DMC 实现，包括 C 和 Rust 语言的实现，这些可以作为理解 DMC 在 C++ 中实现原理的参考。dna-compression 项目 (12) 包含了一个 C 实现 (library/dmc.c)，虽然具体内容在提供的片段中不可见，但其逻辑可以作为 C++ 实现的参考。

## 5. 动态马尔科夫压缩算法 C++ 实现的关键步骤和数据结构解析 (Analysis of Key Steps and Data Structures in C++ Implementation):

   在 DMC 的编码过程中，关键的一步是根据当前的上下文预测下一个比特的概率，并使用算术编码器根据这些概率对实际出现的比特进行编码。编码器首先需要维护当前的上下文，这通常通过一个指向当前马尔科夫模型状态的指针或索引来实现。然后，它会使用当前上下文状态中存储的 0 和 1 的计数来计算下一个比特为 0 或 1 的预测概率。算术编码器会根据这些预测概率更新其内部范围（通常是低位 low 和高位 high），并根据实际出现的比特选择相应的子范围作为新的当前范围。最终，压缩后的比特会从这个范围内提取出来.

   在解码过程中，解码器需要与编码器同步维护相同的马尔科夫模型。这要求解码器也执行相同的初始化、预测、更新和克隆操作。解码器使用当前的编码值（通常表示为一个位于 \[0, 1\) 区间内的浮点数）和模型预测的概率来推断原始的比特值。它通过比较编码值与基于预测概率划分的子范围来确定原始比特是 0 还是 1。然后，解码器会更新其内部范围，使其与编码过程中的范围变化保持一致，从而能够正确地解码后续的比特。

   关键的数据结构在 DMC 的实现中起着至关重要的作用。状态表（或更精确地说，是状态图）用于存储马尔科夫模型的各个状态。每个状态代表一个特定的上下文，并存储了该上下文中 0 和 1 的计数以及指向后续上下文的链接。状态表的每个条目代表一个上下文，并存储了该上下文中 0 和 1 的计数以及指向后续上下文的链接。通常，每个上下文会存储两个指针，一个指向如果下一个比特是 0 时应该转换到的下一个上下文，另一个指向如果下一个比特是 1 时应该转换到的下一个上下文 4。状态表需要能够动态增长以适应新的上下文，这通常通过动态内存分配来实现。计数器用于记录在每个上下文中观察到的 0 和 1 的次数，这些计数直接用于估计下一个比特的概率。计数器的精度（例如使用整数或浮点数）会影响算法的性能和复杂性。

   由于马尔科夫模型的状态数量可能会动态增长，因此有效的内存管理策略对于 DMC 的实现至关重要。通常需要使用动态内存分配和释放的策略，以避免内存泄漏和碎片化。可以使用自定义的内存分配器或智能指针来简化内存管理。为了防止模型无限增长导致内存耗尽，可以考虑限制模型的大小，例如设置最大状态数。当达到限制时，可以停止创建新的状态或使用某种策略（例如最近最少使用）来移除不常用的状态 4。一些实现，如 paq8l，在内存不足时会提高克隆阈值以减缓新状态的创建，或者在极端情况下将模型重置为一个基本的低阶模型 4。

## 6. 不同动态马尔科夫压缩算法的 C++ 实现方法比较 (Comparison of Different C++ Implementation Methods):

   对已知的 DMC 实现进行比较分析可以帮助我们更好地理解其不同的实现方法和特点。

   **Gordon Cormack 的原始 C 代码** (6) 是 DMC 算法的原始实现，由其发明者之一 Gordon Cormack 提供。这份代码是理解 DMC 核心原理的重要参考。原始代码可能更侧重于算法的正确性和原理的验证，而不是极致的性能优化。值得注意的是，原始的实现使用了浮点数进行概率计算，这在某些硬件上可能会影响速度 (17)。此外，由于是较早期的实现，其代码风格和结构可能与现代 C++ 实践有所不同。

   **PAQ8L 中的 DMC 模型** (4) 是一个以实现极高压缩率而闻名的开源压缩器。PAQ8L 内部包含了一个精心设计的 DMC 子模型，但它并非一个独立的 DMC 实现，而是作为其复杂的混合模型的一部分，与其他多种上下文模型相结合，共同进行预测 (4)。PAQ8L 的实现经过了高度优化，可能使用了定点算术或其他低级优化技术来提高速度和效率 (18)。由于其目标是追求极致的压缩率，PAQ8L 的代码复杂度很高，可能难以直接提取和理解其 DMC 部分。

   **GitHub 上的开源项目** (15) 提供了一些 DMC 的实现。例如，oscar-franzen/dynamic-markov-compression (15) 是一个用 Rust 编写的 DMC 实现。虽然不是 C++，但其作者明确提到是作为学习 Rust 的练习，并将原始的 C 代码移植到了 Rust。因此，其实现思路和数据结构设计可以为 C++ 实现提供有价值的参考。另一个相关的项目是 rajatdiptabiswas/dna-compression (12)，该项目专注于 DNA 数据的压缩，其中包含一个 C 语言的 DMC 实现 (library/dmc.c)。尽管具体内容未在提供的材料中给出，但它表明了 DMC 在特定领域的应用尝试，并且其 C 代码可以作为 C++ 实现的参考。

   不同实现方法在压缩率、速度、内存占用和代码复杂性等方面存在差异。原始 C 代码可能在速度上不如高度优化的 PAQ8L，但在理解 DMC 的基本原理上可能更直接和易于上手。PAQ8L 为了追求极致的压缩率，采用了非常复杂的模型和优化策略，因此代码复杂度很高，可能难以直接理解和修改。GitHub 上的开源项目可能各有侧重，例如某些项目可能更注重代码的清晰度和可维护性，而另一些项目可能更关注特定的应用场景（如 DNA 压缩）。它们的性能特点也会有所不同。

   下表总结了不同 DMC 实现方法的关键特点：

### **表 1：不同 DMC 实现方法的比较**

| 实现方法 | 语言 | 压缩率 | 速度 | 内存占用 | 代码复杂性 | 主要特点/备注 |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| Cormack 原始代码 | C | 高 | 中等 | 中等 | 低 | DMC 的原始实现，使用浮点数进行概率计算，代码相对简洁，更侧重于原理验证。 |
| PAQ8L 中的 DMC | C++ | 非常高 | 慢 | 高 | 非常高 | 作为 PAQ8L 复杂混合模型的一部分，经过高度优化，可能使用定点算术，代码非常复杂，难以单独理解。 |
| oscar-franzen | Rust | 高于 gzip | 慢 | 中等 | 中等 | 将原始 C 代码移植到 Rust，作为学习 Rust 的练习，代码清晰度较高，内存安全。 |
| rajatdiptabiswas | C | 未知 | 未知 | 未知 | 未知 | 专注于 DNA 压缩的 DMC 实现，具体性能和代码特点未知。 |

## 7. 开源的动态马尔科夫压缩算法 C++ 库或项目 (Open-Source Dynamic Markov Compression C++ Libraries or Projects):

   在 GitHub 等代码托管平台上搜索 "Dynamic Markov Compression C++" 或 "DMC compression C++" 等关键词，可以找到一些相关的开源项目。例如，oscar-franzen/dynamic-markov-compression (15) 虽然是用 Rust 实现的，但其设计思路和数据结构可以为 C++ 实现提供参考。rajatdiptabiswas/dna-compression (12) 中包含的 C 实现 (library/dmc.c) 也可以作为移植到 C++ 的起点。此外，PAQ 项目 (24) 是开源的，其 DMC 部分的 C++ 代码可以作为研究和学习的资源 (14)。

   评估这些库或项目的成熟度、文档完善程度和社区活跃度需要进一步的考察。通常，可以通过查看项目的提交历史、issue 跟踪系统、pull request 列表以及贡献者的数量来评估项目的活跃程度和维护状态。清晰、详细的文档，包括 API 说明和使用示例，对于用户来说至关重要。代码质量也是一个重要的考虑因素，包括代码的可读性、结构和是否遵循良好的编程实践。

   对于希望使用或贡献 DMC C++ 代码的用户，建议初学者可以从理解原始的 C 代码入手，逐步学习 DMC 的基本原理和实现细节。对于需要高性能和高压缩率的用户，可以深入研究 PAQ8L 的实现，尽管其代码可能较为复杂。积极参与到相关的开源项目中，例如通过贡献代码、报告 bug 或提供改进建议，也是一个很好的学习和贡献方式。

## 8. 总结与展望 (Summary and Future Directions):

   动态马尔科夫压缩算法是一种基于动态构建的马尔科夫模型进行比特预测，并结合算术编码实现高效无损数据压缩的技术 4。它在某些类型的数据上展现出了比传统算法更高的压缩率和优秀的自适应性 1。然而，DMC 的计算复杂度和内存占用相对较高，并且实现难度也较大，这可能限制了其广泛应用 4。

   展望未来，随着计算资源的不断提升，DMC 在对压缩率有极高要求的应用场景中（例如科学数据存储、长期数据归档）可能具有更大的应用潜力。研究方向可以包括探索更高效的马尔科夫模型表示方法，以降低内存占用；使用定点算术或其他优化技术来提高 DMC 的速度；以及开发更易于使用和集成的 DMC C++ 库。此外，针对特定类型的数据（例如基因组数据、时间序列数据）进行 DMC 算法的优化和定制仍然是一个有价值的研究方向。将 DMC 与其他先进的压缩技术相结合，例如神经网络预测模型，也可能在压缩率、速度和内存占用之间取得更好的平衡。尽管 DMC 具有很强的理论压缩能力，但其实际应用受到了一些限制。未来的研究应该侧重于使其更高效和易于使用，以充分发挥其在各种数据压缩应用中的潜力。

## **引用的著作**

（访问时间为：2025年4月8日）

1. Data Compression Using Dynamic Markov Modelling \- Computer Science,  [https://webhome.cs.uvic.ca/\~nigelh/Publications/DMC.pdf](https://webhome.cs.uvic.ca/~nigelh/Publications/DMC.pdf)
2. Data Compression Using Dynamic Markov Modelling,  [https://academic.oup.com/comjnl/article-pdf/30/6/541/935458/30-6-541.pdf](https://academic.oup.com/comjnl/article-pdf/30/6/541/935458/30-6-541.pdf)
3. DATA COMPRESSION USING DYNAMIC MARKOV MODELLING \- CiteSeerX,  [https://citeseerx.ist.psu.edu/document?repid=rep1\&type=pdf\&doi=65a5e1c2f569d24634cb5f7df3926dde4102fa3f](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=65a5e1c2f569d24634cb5f7df3926dde4102fa3f)
4. Dynamic Markov compression \- Wikipedia,  [https://en.wikipedia.org/wiki/Dynamic\_Markov\_compression](https://en.wikipedia.org/wiki/Dynamic_Markov_compression)
5. Data Compression Using Dynamic Markov Modelling | The Computer Journal,  [https://academic.oup.com/comjnl/article-abstract/30/6/541/327619](https://academic.oup.com/comjnl/article-abstract/30/6/541/327619)
6. Dynamic Markov Compression,  [https://go-compression.github.io/algorithms/dmc/](https://go-compression.github.io/algorithms/dmc/)
7. Markov chain \- Wikipedia,  [https://en.wikipedia.org/wiki/Markov\_chain](https://en.wikipedia.org/wiki/Markov_chain)
8. Markov model \- Wikipedia,  [https://en.wikipedia.org/wiki/Markov\_model](https://en.wikipedia.org/wiki/Markov_model)
9. Markov Model \- What Is It, Examples, Applications, Advantages \- WallStreetMojo,  [https://www.wallstreetmojo.com/markov-model/](https://www.wallstreetmojo.com/markov-model/)
10. Data Compression/Markov models \- Wikibooks, open books for an open world,  [https://en.wikibooks.org/wiki/Data\_Compression/Markov\_models](https://en.wikibooks.org/wiki/Data_Compression/Markov_models)
11. Data Compression Using Dynamic Markov Modelling | The Computer Journal,  [https://academic.oup.com/comjnl/article/30/6/541/327619](https://academic.oup.com/comjnl/article/30/6/541/327619)
12. rajatdiptabiswas/dna-compression: Analyzing compression algorithms for genomic sequencing data \- GitHub,  [https://github.com/rajatdiptabiswas/dna-compression](https://github.com/rajatdiptabiswas/dna-compression)
13. DMC(Dynamic Markov Compression) \- CS@UCF,  [http://www.cs.ucf.edu/courses/cap5015/DMC.pdf](http://www.cs.ucf.edu/courses/cap5015/DMC.pdf)
14. paq/paq8l/paq8l.cpp at master · JohannesBuchner/paq · GitHub,  [https://github.com/JohannesBuchner/paq/blob/master/paq8l/paq8l.cpp](https://github.com/JohannesBuchner/paq/blob/master/paq8l/paq8l.cpp)
15. oscar-franzen/dynamic-markov-compression: Dynamic Markov Compression (DMC) implemented in Rust. DMC is a lossless, general purpose, compression technique based on a probabilistic model. \- GitHub,  [https://github.com/oscar-franzen/dynamic-markov-compression](https://github.com/oscar-franzen/dynamic-markov-compression)
16. An exploration of dynamic Markov compression \- University of Canterbury,  [https://ir.canterbury.ac.nz/bitstream/10092/9572/1/whitehead\_thesis.pdf](https://ir.canterbury.ac.nz/bitstream/10092/9572/1/whitehead_thesis.pdf)
17. DMC,  [https://www.jjj.de/crs4/dmc.c](https://www.jjj.de/crs4/dmc.c)
18. ocamyd \-.:: GEOCITIES.ws ::.,  [http://www.geocities.ws/ocamyd/](http://www.geocities.ws/ocamyd/)
19. Dynamic Markov compression \- Acemap,  [https://ddescholar.acemap.info/field/2024503987](https://ddescholar.acemap.info/field/2024503987)
20. arithc.c,  [https://www.cs.cmu.edu/afs/andrew/scs/cs/15-381/archive/OldFiles/lib/cgi-bin/.g/doc/.g/scottd/arithcoder/dmc/arithc.c](https://www.cs.cmu.edu/afs/andrew/scs/cs/15-381/archive/OldFiles/lib/cgi-bin/.g/doc/.g/scottd/arithcoder/dmc/arithc.c)
21. Data Compression Models for Prediction and Classification \- PLG,  [https://plg.uwaterloo.ca/\~gvcormac/cormack-nato.pdf](https://plg.uwaterloo.ca/~gvcormac/cormack-nato.pdf)
22. CS Exposed IV Big Questions, Easy Answers? \- Gordon V. Cormack,  [https://cormack.uwaterloo.ca/\~gvcormac/csexposed.pdf](https://cormack.uwaterloo.ca/~gvcormac/csexposed.pdf)
23. Information retrieval : implementing and evaluating search engines \- WordPress.com,  [https://mitmecsept.files.wordpress.com/2018/05/stefan-bc3bcttcher-charles-l-a-clarke-gordon-v-cormack-information-retrieval-implementing-and-evaluating-search-engines-2010-mit.pdf](https://mitmecsept.files.wordpress.com/2018/05/stefan-bc3bcttcher-charles-l-a-clarke-gordon-v-cormack-information-retrieval-implementing-and-evaluating-search-engines-2010-mit.pdf)
24. The PAQ Data Compression Programs \- Matt Mahoney,  [https://www.mattmahoney.net/dc/paq.html](https://www.mattmahoney.net/dc/paq.html)
25. paq/paq8px\_v67/paq8px.cpp at master · JohannesBuchner/paq \- GitHub,  [https://github.com/JohannesBuchner/paq/blob/master/paq8px\_v67/paq8px.cpp](https://github.com/JohannesBuchner/paq/blob/master/paq8px_v67/paq8px.cpp)
26. Hacker News \- RSSing.com,  [https://hacker1976.rssing.com/chan-69386952/all\_p386.html](https://hacker1976.rssing.com/chan-69386952/all_p386.html)
27. Data Compression Explained \- Matt Mahoney,  [https://mattmahoney.net/dc/dce.html](https://mattmahoney.net/dc/dce.html)
28. Crinkler secrets, 4k intro executable compressor at its best \- code4k,  [http://code4k.blogspot.com/2010/12/crinkler-secrets-4k-intro-executable.html](http://code4k.blogspot.com/2010/12/crinkler-secrets-4k-intro-executable.html)
29. PAQ \- Wikipedia,  [https://en.wikipedia.org/wiki/PAQ](https://en.wikipedia.org/wiki/PAQ)
