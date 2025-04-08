# **LZSS 无损压缩算法原理与实现**

## 概述

* 来自：[Gemini 2.5 Pro](https://gemini.google.com/)
* 网址：[Gemini Deep Research](https://gemini.google.com/app/6bf5987f05d09349?hl=zh-cn)
* Google Docs：[Google 文档](https://docs.google.com/document/d/1WevJwNJQQopz3A3_S7CNZeY5F2-MtM9QJIIN-tDCC3A/edit?tab=t.0)

## **1. LZSS 压缩算法概述**

在信息技术领域，数据压缩是一项至关重要的技术，旨在减少数据存储空间需求和加快数据传输速度。无损数据压缩是其中的一个重要分支，它能够在不丢失任何原始信息的前提下缩小数据体积，保证数据可以被完整地恢复。LZSS（Lempel-Ziv-Storer-Szymanski）算法正是一种被广泛应用的无损数据压缩技术。

LZSS 算法属于字典编码的一种方法。与试图通过减少表示单个符号所需的平均比特数（如霍夫曼编码）不同，字典编码通过将重复出现的符号序列替换为指向字典中该序列位置的引用来实现压缩。字典编码的核心思想在于识别数据中的冗余模式，并用更短的“代码”来表示这些模式，从而达到压缩的目的。LZSS 算法的关键特性在于其使用动态构建的字典，该字典基于输入数据流中已经出现过的符号序列。

LZSS 算法由 James A. Storer 和 Thomas Szymanski 于 1982 年提出，是 LZ77 算法的一个重要变种。理解 LZSS 与 LZ77 之间的关系有助于认识 LZSS 的设计思想。LZ77 是 Lempel 和 Ziv 在 1977 年提出的最早的滑动窗口压缩算法。LZSS 在 LZ77 的基础上进行了改进，主要区别在于 LZSS 在某些情况下会选择不进行替换，以避免压缩后的数据比原始数据更大。这种策略使得 LZSS 在处理某些类型的数据时更加有效。

LZSS算法因其相对简单、易于实现以及高效的解压缩速度而被广泛应用于各种领域。许多流行的归档工具，如 ARJ、RAR、ZOO 和 LHarc，都将 LZSS 作为主要的压缩算法。此外，苹果公司的 macOS 操作系统也使用 LZSS 作为内核代码的压缩方法之一。在游戏开发领域，例如 Game Boy Advance BIOS，也采用了轻微修改的 LZSS 格式。其作为许多广泛使用的压缩器（如 Deflate）的核心算法，进一步体现了其重要性。LZSS 在解压缩速度方面的优势，使其在对速度要求较高的应用场景中尤为适用。

## **2. LZSS 算法的核心原理**

LZSS 算法的核心在于利用一个动态维护的“字典”来压缩数据，这个字典实际上是一个在输入数据流上滑动的窗口。该滑动窗口记录了最近处理过的数据，并充当查找重复模式的参考。与一些使用静态字典的压缩方法不同，LZSS 的字典是随着压缩过程动态变化的，使其能够适应数据中不断变化的局部冗余。

LZSS 算法将输入数据编码为两种基本类型：**字面量（Literal）和指针（Pointer）**。字面量是指那些在滑动窗口中没有找到匹配的原始数据字节，它们被直接写入压缩后的数据流。指针则用于表示在滑动窗口中找到的重复字符串。一个指针通常包含两个信息：**偏移量（Offset）和长度（Length）**。偏移量指示了匹配字符串在滑动窗口中相对于当前位置的距离，而长度则表示了匹配字符串的字符数。

LZSS 算法引入了一个重要的概念，即“**盈亏平衡点（Break-Even Point）**”或称为最小匹配长度。只有当在滑动窗口中找到的匹配长度超过这个阈值时，使用指针进行编码才是有利的。这是因为编码一个指针本身需要一定的存储空间（通常是几个字节，包括标志位、偏移量和长度），如果匹配的字符串太短，那么指针所占用的空间可能比直接存储原始字符串还要大，导致压缩率下降甚至数据膨胀。因此，LZSS 算法会在匹配长度小于该阈值时选择直接输出原始字符。通常，这个最小匹配长度被设置为 2 或 3 个字符。

为了让解码器能够区分压缩数据流中的字面量和指针，LZSS 算法在每个编码单元之前使用 **标志位（Flag）** 进行标识。通常，一个标志位（可能是单个比特，也可能包含在控制字节中）用于指示接下来的数据是一个原始字节还是一个指向先前出现字符串的指针。这种机制使得解码器能够正确地解析压缩后的数据并恢复原始信息。

## **3. LZSS 算法的详细工作流程**

### **3.1 LZSS 压缩流程**

LZSS 的压缩过程是一个迭代的过程，它通过在滑动窗口中搜索与当前输入数据匹配的最长字符串，并根据匹配情况输出相应的编码。

其详细步骤如下：

1. **初始化滑动窗口**：压缩开始时，需要将滑动窗口初始化为一个已知的状态，通常填充一些特定的字符，如空格或空字符。
2. **读取先行缓冲区**：从输入数据流中读取一段固定长度的数据到先行缓冲区（Lookahead Buffer）中。先行缓冲区的长度通常设置为允许的最大匹配长度。
3. **在滑动窗口中搜索最长匹配**：在滑动窗口的搜索缓冲区（Search Buffer，滑动窗口中已编码的部分）中，查找与先行缓冲区起始位置的最长字符串相匹配的子串。搜索过程通常采用贪婪策略，即总是尝试找到当前位置的最长匹配。
4. **编码匹配或输出字面量**：根据步骤 3 的搜索结果，进行编码：
   * **如果找到长度大于或等于最小匹配长度的匹配**：输出一个表示“已编码”的标志位，然后输出匹配字符串在滑动窗口中的偏移量（距离当前位置的后向距离）和匹配的长度。
   * **如果没有找到符合条件的匹配**：输出一个表示“未编码”的标志位，然后直接输出先行缓冲区的第一个字符（作为字面量）。
5. **更新滑动窗口**：无论是否找到匹配，都需要更新滑动窗口。将刚刚编码输出的数据（无论是匹配的字符串还是字面量）添加到滑动窗口中。由于滑动窗口的大小是固定的，这可能涉及到移除窗口中最旧的数据，为新数据腾出空间，从而实现“滑动”的效果。
6. **读取更多输入**：根据步骤 4 中输出的字符数（如果是匹配，则读取匹配长度的字符；如果是字面量，则读取一个字符），从输入数据流中读取相应数量的新字符到先行缓冲区，以供下一轮匹配。
7. **迭代**：重复步骤 3 到步骤 6，直到整个输入数据流都被处理完毕。

### **3.2 LZSS 解压缩流程**

LZSS 的解压缩过程与压缩过程相对应，它利用压缩数据流中的标志位、偏移量和长度信息来重建原始数据。

其详细步骤如下：

1. **初始化滑动窗口**：与压缩过程一样，解压缩开始时也需要将滑动窗口初始化为与压缩器相同的已知状态。
2. **读取标志位**：从压缩数据流中读取一个标志位，判断接下来的数据是已编码的字符串还是未编码的字面量。
3. **解码**：根据标志位的不同，执行相应的解码操作：
   * **如果标志位指示已编码的字符串**：从压缩数据流中读取偏移量和长度信息。然后，根据偏移量在滑动窗口中找到对应的位置，并将从该位置开始的指定长度的字符串复制到解压缩后的输出中。
   * **如果标志位指示未编码的字面量**：从压缩数据流中读取下一个字符，并将其直接写入解压缩后的输出。
4. **更新滑动窗口**：将刚刚解码输出的数据（无论是从滑动窗口复制的字符串还是直接读取的字面量）添加到滑动窗口中，以保持与压缩器滑动窗口状态的同步。
5. **迭代**：重复步骤 2 到步骤 4，直到整个压缩数据流都被处理完毕，通常会遇到一个特定的结束标志或输入流的结束。

值得注意的是，LZSS 的解压缩过程通常比压缩过程在资源上更不密集，因为它不需要搜索匹配的字符串，只需要根据偏移量和长度进行复制即可。

## **4. 滑动窗口和先行缓冲区在LZSS中的作用与实现**

### **4.1 滑动窗口（字典）**

滑动窗口在 LZSS 算法中扮演着至关重要的角色，它充当着一个动态的“字典”，存储了最近处理过的数据。

其主要作用在于：

* **提供历史数据参考**：滑动窗口记录了输入流中已经出现过的符号序列，使得算法能够检测并编码重复出现的模式。
* **动态适应数据特性**：由于滑动窗口随着数据的处理不断向前滑动，它能够反映数据流最近的局部特性，从而更有效地压缩具有局部重复性的数据。

滑动窗口的大小是一个关键参数，通常用 N 表示。滑动窗口的大小直接影响着算法的性能：

* **搜索时间**：较大的滑动窗口意味着在压缩过程中需要搜索更大的范围来寻找匹配，这会增加搜索时间。
* **偏移量大小**：偏移量用于指示匹配字符串在滑动窗口中的位置。滑动窗口越大，表示偏移量所需的比特数就越多。
* **匹配可能性**：较大的滑动窗口能够存储更多的历史数据，从而增加了找到更长匹配的可能性，这通常有助于提高压缩率。

滑动窗口通常以 **循环缓冲区（Circular Buffer或Ring Buffer）** 的形式实现。循环缓冲区的优点在于能够高效地添加新数据并覆盖旧数据，模拟窗口滑动的效果。当新的数据被处理并需要加入滑动窗口时，它会覆盖缓冲区中最早的数据，从而保持窗口大小的固定。当然，也存在其他的窗口管理方法，例如使用链表或树结构，这些方法可能在搜索和更新性能上有所不同。

### **4.2 先行缓冲区**

先行缓冲区（Lookahead Buffer）是滑动窗口的另一个重要组成部分，它保存了输入数据流中紧随当前编码位置之后的一段固定长度的数据。

其主要作用是：

* **提供待匹配的数据**：压缩算法在滑动窗口的搜索缓冲区中查找与先行缓冲区起始部分相匹配的字符串。
* **决定最大匹配长度**：先行缓冲区的大小（通常用F表示）限制了算法能够找到并编码的最大匹配长度。

先行缓冲区的大小也会影响压缩性能：

* **更长的匹配**：较大的先行缓冲区允许算法尝试匹配更长的字符串序列，如果数据中存在较长的重复模式，这有助于提高压缩率。
* **复杂性**：与滑动窗口类似，较大的先行缓冲区也可能增加匹配搜索的复杂性。

压缩算法通过比较先行缓冲区的内容与滑动窗口的搜索缓冲区的内容，来找到最长的匹配序列。**编码位置（Coding Position）** 指的是先行缓冲区在输入数据流中的起始位置，即当前正在考虑进行压缩的数据的起点。

## **5. LZSS 算法的匹配查找与编码**

### **5.1 贪婪匹配算法**

LZSS 算法通常采用 **贪婪匹配（Greedy Matching）** 策略。这意味着在每个编码步骤，算法都会尝试在滑动窗口的搜索缓冲区中找到与先行缓冲区起始位置尽可能长的匹配字符串。一旦找到最长的匹配（或者没有找到长度超过阈值的匹配），算法就会立即对该结果进行编码并继续处理下一个位置。虽然贪婪匹配实现简单且效率较高，但它并不保证获得全局最优的压缩效果，因为在当前位置选择一个较短的匹配有时可能为后续找到更长的匹配创造机会。

### **5.2 字符串匹配技术**

为了在滑动窗口中高效地查找最长匹配，可以使用多种字符串匹配技术：

* **线性搜索（Brute Force）**：最简单的方法，将先行缓冲区中的字符串与滑动窗口中的每个可能子串进行比较。这种方法实现简单，但效率较低，尤其是在滑动窗口较大时。
* **哈希表（Hash Table）**：可以使用哈希表来快速定位滑动窗口中可能匹配的起始位置。通过对先行缓冲区的起始几个字符计算哈希值，可以在哈希表中快速找到滑动窗口中具有相同哈希值的子串，然后进行详细比较。
* **二叉搜索树（Binary Search Tree）**：二叉搜索树可以用于存储滑动窗口中的所有子串，并支持高效的查找操作。一些 LZSS 实现，如 Haruhiko Okumura 的实现，就使用了二叉搜索树来加速最长匹配的搜索。
* **后缀树（Suffix Tree）**：后缀树是一种更高级的数据结构，可以非常高效地找到一个字符串在另一个字符串中的所有出现位置，以及最长公共前缀等信息。虽然实现较为复杂，但可以提供很高的匹配效率。

选择哪种字符串匹配技术需要在实现复杂度和搜索效率之间进行权衡。例如，使用哈希表或二叉搜索树通常可以显著提高搜索速度，但也会增加实现的复杂性。

### **5.3 编码匹配的字符串**

一旦在滑动窗口中找到长度大于或等于最小匹配长度的匹配字符串，LZSS 就会将其编码为一个指针。这个指针通常包含以下两个部分：

* **偏移量（Offset 或 Distance）**：表示匹配字符串在滑动窗口中相对于当前编码位置的后向距离。偏移量的大小通常需要足够的比特数来表示整个滑动窗口的范围。例如，如果滑动窗口大小为 4096 字节，则偏移量可能需要 12 位（因为 2 ^ 12 = 4096）。
* **长度（Length 或 Run Length）**：表示匹配字符串的字符数。长度的大小通常需要足够的比特数来表示允许的最大匹配长度。例如，如果最大匹配长度为16，则长度可能需要4位（因为 2 ^ 4 = 16）。

在实际编码时，偏移量和长度会被组合成一个或多个字节。例如，在 Bohemia Interactive 的 LZSS 实现中，一个 16 位的指针包含 12 位的偏移量和 4 位的长度。

### **5.4 编码未匹配的字符**

如果算法在滑动窗口中没有找到长度超过最小匹配长度的匹配，那么先行缓冲区的第一个字符（或几个字符，取决于最小匹配长度）将作为 **字面量（Literal）** 直接编码输出。字面量通常就是原始的字节值。

### **5.5 标志位编码方案**

为了区分压缩数据流中的字面量和指针，LZSS 需要在每个编码单元之前添加标志位。

常见的标志位编码方案包括：

* **单比特标志**：为每个编码单元（无论是字面量还是指针）添加一个前导比特。例如，'0' 可能表示下一个是字面量字节，而 '1' 表示下一个是指针（偏移量和长度）。
* **控制字节**：一些实现会将多个标志位组合到一个控制字节中。例如，一个 8 位的控制字节可以为接下来的 8 个编码单元分别指示其类型。控制字节中的每个比特对应一个编码单元，'1' 可能表示字面量，'0' 表示指针。

不同的标志位编码方案会影响压缩数据的开销和解码的效率。例如，使用控制字节可以在一定程度上减少标志位的总数，但需要在解码时处理整个字节。

## **6. LZSS 算法的解压缩过程详解**

LZSS 的解压缩过程是压缩过程的逆向操作，它利用压缩数据流中的标志位、偏移量和长度信息来重建原始数据。

1. **重建滑动窗口**：解压缩器也需要维护一个与压缩器大小和初始状态相同的滑动窗口（字典）。随着解压缩的进行，解压缩器会动态地更新这个窗口，使其与压缩器在压缩相同数据时窗口的状态保持一致。
2. **处理标志位以识别数据类型**：解压缩器首先从压缩数据流中读取标志位（或控制字节中的一个比特）。这个标志位指示了接下来的数据是字面量还是指针。
3. **解码字面量**：如果标志位指示是字面量，解压缩器直接从压缩数据流中读取下一个字节（或多个字节，取决于实现）作为原始字符，并将其写入解压缩后的输出。同时，这个字面量字节也会被添加到解压缩器的滑动窗口中的当前位置。
4. **解码指针（偏移量和长度）**：如果标志位指示是指针，解压缩器会从压缩数据流中读取后续的比特来获取偏移量和长度信息。读取的比特数取决于压缩时使用的滑动窗口大小和最大匹配长度。
5. **从字典中复制**：解压缩器使用获取到的偏移量，在当前的滑动窗口中回溯相应的距离。然后，它从回溯的位置开始，复制长度所指定的字节数到解压缩后的输出。关键的是，这些被复制的字节也会被添加到解压缩器的滑动窗口中的当前位置，这样就保证了滑动窗口的内容与压缩过程中相同位置的内容一致。值得注意的是，匹配的长度可能超过当前输出缓冲区的大小，这允许有效地编码重复序列。
6. **处理结束标记**：解压缩器需要识别压缩数据流的结束。这可以通过特定的结束标记值（例如，在 Texas Instruments 的实现中，偏移量为特定的结束数据值）或者简单地到达输入流的末尾来判断。
7. **解压缩过程中的错误处理**：在实际应用中，解压缩器可能需要处理一些错误情况，例如遇到无效的偏移量（指向滑动窗口之外的位置）或格式错误的压缩数据流。

## **7. LZSS 的 C++ 实现示例分析**

存在多种 LZSS 算法的 C++ 实现，从简单的教学示例到高度优化的库都有。一些实现是直接基于 Haruhiko Okumura 最初的 C 代码进行改编的。

### **7.1 简单 C++ 实现分析（以 Jeremy Collake 的实现为例）**

该实现定义了滑动窗口大小为 4095 字节（需要 12 位偏移量）和最大匹配长度为 17 字节（长度编码为长度减 2，需要 4 位）。编码后的匹配表示为一个 16 位的码字，包含偏移量和长度。控制位以每 8 个一组的方式存储。

* **数据结构**：主要使用字符数组来存储源数据和目标数据。MAX\_WINDOWSIZE 和 MAX\_LENGTH 等常量定义了滑动窗口和最大匹配长度。
* **编码函数逻辑 (CompressData)**：该函数遍历源数据，使用 SearchForPhrase 函数在滑动窗口中查找最长匹配。它实现了惰性求值（Lazy Evaluation）来寻找潜在的更优匹配。根据找到的匹配情况，将数据编码为字面量字节或包含偏移量和长度的 16 位码字。使用控制字节来指示后续数据的类型。
* **解码函数逻辑 (DecompressData)**：该函数读取控制字节，根据控制位判断后续是字面量还是码字。如果是码字，则从中提取偏移量和长度，并在目标缓冲区（充当滑动窗口）中回溯偏移量，复制长度个字节到当前位置，从而重建原始数据。
* **关键参数**：窗口大小为 4095，最大匹配长度为 17，最小匹配长度隐含为 2。

### **7.2 Haruhiko Okumura 的 LZSS 实现（C 语言，但对 C++ 实现有重要影响）**

Okumura 的实现（最初使用 C 语言）以其高效性而闻名，它使用 **二叉搜索树** 来加速滑动窗口中最长匹配的搜索。这种方法比简单的线性搜索效率更高，尤其是在滑动窗口较大时。通常，该实现使用数组来模拟二叉搜索树的节点和指针，这在某些情况下可以提高内存效率。

### **7.3 其他 C/C++ 库和代码仓库**

除了上述示例，还有其他的 C/C++ LZSS 实现可供参考，例如 Michael Dipperstein 提供的 ANSI C 库，该库支持多种可插拔的字符串匹配算法（包括暴力搜索、哈希表、KMP 算法、链表和二叉搜索树）。此外，Allegro 库也包含 LZSS 的实现，可以作为参考。Matt Seabrook 对 Okumura 的原始 C 代码进行了重构，并提供了 C++ 版本（虽然标记为旧版本，但仍可作为学习资源）。

## **8. 不同 C++ LZSS 实现的比较分析**

不同的 C++ LZSS 实现可能在多个方面存在差异：

* **性能**：压缩率和压缩/解压缩速度是衡量实现性能的关键指标。不同的实现由于采用不同的字符串匹配算法、滑动窗口和先行缓冲区的大小以及优化策略，其性能表现会有所不同。通常，更复杂的匹配算法（如使用二叉搜索树或后缀树）可能会带来更高的压缩率，但也会增加压缩时间。解压缩速度通常是 LZSS 的优势。
* **代码复杂性和可读性**：一些实现可能为了追求更高的性能而牺牲了代码的可读性和可维护性，而另一些实现则更注重代码的清晰度和易于理解。
* **LZSS格式和参数的差异**：不同的实现可能采用略有不同的 LZSS 压缩格式，例如标志位的编码方式、偏移量和长度所占用的比特数、以及滑动窗口和先行缓冲区的默认大小。这可能导致不同实现之间压缩的数据无法互相兼容。
* **适用场景**：不同的实现可能更适合特定的应用场景。例如，对于需要快速解压缩的游戏应用，可能更倾向于选择解压缩速度快的实现，即使压缩率稍低；而对于存储应用，可能更关注压缩率。
* **并行化和硬件加速的潜力**：一些研究和实现致力于将 LZSS 算法并行化，以利用多核处理器或 GPU 来提高压缩和解压缩的速度，尤其是在处理大数据时。

为了更清晰地对比 LZ77 和 LZSS，我们可以总结如下表：

### **表1：LZ77 和 LZSS 的比较**

| 特性 | LZ77 | LZSS |
| :---- | :---- | :---- |
| 处理短匹配 | 有时会输出比原始数据更长的引用 | 对于短于盈亏平衡点的匹配不进行替换 |
| 输出 | \<偏移量, 长度, 下一个符号\> | \<标志位, (偏移量, 长度) 或 字面量\> |
| 标志位 | 不总是作为专用比特存在 | 使用显式的标志位 |

以下是 Jeremy Collake 的简单 LZSS 实现的参数总结：

### **表2：Jeremy Collake 的简单 LZSS 实现的参数**

| 参数 | 值 |
| :---- | :---- |
| 滑动窗口大小 | 4095字节（偏移量需要12位） |
| 最大匹配长度 | 17字节（长度编码需要4位） |
| 最小匹配长度 | 2字节（隐含） |
| 码字大小 | 16位（偏移量 \+ 长度） |
| 控制位 | 每8个一组 |

## **9. 结论**

LZSS 算法作为一种经典的无损数据压缩技术，其核心在于利用滑动窗口维护历史数据，并通过查找和编码重复出现的字符串来达到压缩的目的。它通过引入字面量和指针的概念，以及使用标志位来区分它们，实现了数据的有效压缩和无损恢复。

LZSS 算法因其相对简单和高效的解压缩速度，在各种领域都得到了广泛应用。不同的 C++ 实现展示了在性能、复杂性和适用性方面的权衡。开发者在选择或实现 LZSS 算法时，需要根据具体的应用场景和需求，例如对压缩率和速度的要求，来选择合适的实现策略和参数配置。

随着计算技术的不断发展，对数据压缩算法的性能要求也越来越高。未来的研究可能会继续探索 LZSS 算法的并行化和硬件加速，以及与其他压缩技术的结合，以进一步提高其压缩效率和处理能力。

## **引用的著作**

（访问时间为：2025 年 4 月 8 日）

1. LZSS Quine Puzzle, [https://nullprogram.com/blog/2014/11/22/](https://nullprogram.com/blog/2014/11/22/)
2. LZSS Encoding Example | Download Scientific Diagram, [https://www.researchgate.net/figure/LZSS-Encoding-Example\_fig1\_224264427](https://www.researchgate.net/figure/LZSS-Encoding-Example_fig1_224264427)
3. LZSS (LZ77) Discussion and Implementation, [https://michaeldipperstein.github.io/lzss.html](https://michaeldipperstein.github.io/lzss.html)
4. Lempel–Ziv–Storer–Szymanski \- Wikipedia, [https://en.wikipedia.org/wiki/Lempel%E2%80%93Ziv%E2%80%93Storer%E2%80%93Szymanski](https://en.wikipedia.org/wiki/Lempel%E2%80%93Ziv%E2%80%93Storer%E2%80%93Szymanski)
5. Parallel Lossless Data Compression Algorithms \- GitHub Pages, [https://hzxa21.github.io/15618-FinalProject/](https://hzxa21.github.io/15618-FinalProject/)
6. Exploring the LZ77 Algorithm \- Coder Spirit, [https://blog.coderspirit.xyz/blog/2023/06/04/exploring-the-lz77-algorithm/](https://blog.coderspirit.xyz/blog/2023/06/04/exploring-the-lz77-algorithm/)
7. LZSS/README.md at main · mattseabrook/LZSS \- GitHub, [https://github.com/mattseabrook/LZSS/blob/main/README.md](https://github.com/mattseabrook/LZSS/blob/main/README.md)
8. LZSS \- The Hitchhiker's Guide to Compression, [https://go-compression.github.io/algorithms/lzss/](https://go-compression.github.io/algorithms/lzss/)
9. LZSS, [https://crosswire.org/jsword/javadoc/org/crosswire/common/compress/LZSS.html](https://crosswire.org/jsword/javadoc/org/crosswire/common/compress/LZSS.html)
10. Venomalia/AuroraLib.Compression: Supports a wide range of compression algorithms mainly used in video games, like LZSS, LZ10, LZ11, MIO0, YAZ0, YAY0, PRS and more. \- GitHub, [https://github.com/Venomalia/AuroraLib.Compression](https://github.com/Venomalia/AuroraLib.Compression)
11. An example of LZSS algorithm. The left is original data, and the right... \- ResearchGate, [https://www.researchgate.net/figure/An-example-of-LZSS-algorithm-The-left-is-original-data-and-the-right-is-compressed\_fig1\_370070645](https://www.researchgate.net/figure/An-example-of-LZSS-algorithm-The-left-is-original-data-and-the-right-is-compressed_fig1_370070645)
12. honza-kasik/lzss: This is naive\* implementation of LZSS compression algorithm \- GitHub, [https://github.com/honza-kasik/lzss](https://github.com/honza-kasik/lzss)
13. LZSS compression \- ModdingWiki, [https://moddingwiki.shikadi.net/wiki/LZSS\_compression](https://moddingwiki.shikadi.net/wiki/LZSS_compression)
14. The LZSS algorithm \- C++ Forum \- Cplusplus, [https://cplusplus.com/forum/general/17012/](https://cplusplus.com/forum/general/17012/)
15. Compressed LZSS File Format \- Bohemia Interactive Community, [https://community.bistudio.com/wiki/Compressed\_LZSS\_File\_Format](https://community.bistudio.com/wiki/Compressed_LZSS_File_Format)
16. Are both of these algorithms valid implementations of LZSS? \- Stack Overflow, [https://stackoverflow.com/questions/32214469/are-both-of-these-algorithms-valid-implementations-of-lzss](https://stackoverflow.com/questions/32214469/are-both-of-these-algorithms-valid-implementations-of-lzss)
17. LZ77 compression in Javascript. When I was working on a library for ..., [https://medium.com/@vincentcorbee/lz77-compression-in-javascript-cd2583d2a8bd](https://medium.com/@vincentcorbee/lz77-compression-in-javascript-cd2583d2a8bd)
18. Still can't understand the LZSS algorith \- C++ Forum \- Cplusplus, [https://cplusplus.com/forum/general/18517/](https://cplusplus.com/forum/general/18517/)
19. clownlzss – A 'Perfect' LZSS Compressor \- Clownacy's Corner, [https://clownacy.wordpress.com/2021/10/14/clownlzss-a-perfect-lzss-compressor/](https://clownacy.wordpress.com/2021/10/14/clownlzss-a-perfect-lzss-compressor/)
20. downloads.ti.com, [https://downloads.ti.com/docs/esd/SPRU514/lempel-ziv-storer-szymanski-compression-lzss-format-stdz0543083.html](https://downloads.ti.com/docs/esd/SPRU514/lempel-ziv-storer-szymanski-compression-lzss-format-stdz0543083.html)
21. LZSS Implementation in C++, [https://bitsum.com/files/lzss\_cpp.html](https://bitsum.com/files/lzss_cpp.html)
22. MichaelDipperstein/lzss: lzss: An ANSI C implementation of the LZSS compression algorithm \- GitHub, [https://github.com/MichaelDipperstein/lzss](https://github.com/MichaelDipperstein/lzss)
23. How the LZ77 compression algorithm handles the case when the entire look-ahead buffer is matched in the search buffer \- Computer Science Stack Exchange, [https://cs.stackexchange.com/questions/75925/how-the-lz77-compression-algorithm-handles-the-case-when-the-entire-look-ahead-b](https://cs.stackexchange.com/questions/75925/how-the-lz77-compression-algorithm-handles-the-case-when-the-entire-look-ahead-b)
24. Matches overlapping lookahead on LZ77/LZSS with suffix trees \- Stack Overflow, [https://stackoverflow.com/questions/31347593/matches-overlapping-lookahead-on-lz77-lzss-with-suffix-trees](https://stackoverflow.com/questions/31347593/matches-overlapping-lookahead-on-lz77-lzss-with-suffix-trees)
25. Confusing about LZSS algorithm... \- C Board, [https://cboard.cprogramming.com/c-programming/132971-confusing-about-lzss-algorithm.html](https://cboard.cprogramming.com/c-programming/132971-confusing-about-lzss-algorithm.html)
26. Searching for fast C++ LZSS decompression implementation \- GameDev.net, [https://www.gamedev.net/forums/topic/418824-searching-for-fast-c-lzss-decompression-implementation/](https://www.gamedev.net/forums/topic/418824-searching-for-fast-c-lzss-decompression-implementation/)
27. lzss/README at master · MichaelDipperstein/lzss \- GitHub, [https://github.com/MichaelDipperstein/lzss/blob/master/README](https://github.com/MichaelDipperstein/lzss/blob/master/README)
28. Serial LZSS, parallel LZSS, and CUDA LZSS comparison 4\) Comparison with GZIP and ZLIB \- ResearchGate, [https://www.researchgate.net/figure/Serial-LZSS-parallel-LZSS-and-CUDA-LZSS-comparison-4-Comparison-with-GZIP-and-ZLIB\_fig3\_258234899](https://www.researchgate.net/figure/Serial-LZSS-parallel-LZSS-and-CUDA-LZSS-comparison-4-Comparison-with-GZIP-and-ZLIB_fig3_258234899)
29. Searching for fast C++ LZSS decompression implementation \- GameDev.net, [https://www.gamedev.net/forums/topic/418824-searching-for-fast-c-lzss-decompression-implementation/?page=2](https://www.gamedev.net/forums/topic/418824-searching-for-fast-c-lzss-decompression-implementation/?page=2)
30. LZSS Decompression \- help \- The Rust Programming Language Forum, [https://users.rust-lang.org/t/lzss-decompression/39920](https://users.rust-lang.org/t/lzss-decompression/39920)
31. Latency-aware adaptive micro-batching techniques for streamed data compression on graphics processing units \- uniPi, [https://pages.di.unipi.it/mencagli/downloads/Preprint-CCPE-2020.pdf](https://pages.di.unipi.it/mencagli/downloads/Preprint-CCPE-2020.pdf)
32. MASSIVELY PARALLEL LZ77 COMPRESSION AND DECOMPRESSION ON THE GPU by Kayla Wesley, BS \- TXST Digital Repository, [https://digital.library.txst.edu/server/api/core/bitstreams/5421889e-b574-403b-8851-63396919e344/content](https://digital.library.txst.edu/server/api/core/bitstreams/5421889e-b574-403b-8851-63396919e344/content)
33. LZSS Algorithm Pipeline Architecture. \- Forum for Electronics, [https://www.edaboard.com/threads/lzss-algorithm-pipeline-architecture.313078/](https://www.edaboard.com/threads/lzss-algorithm-pipeline-architecture.313078/)
