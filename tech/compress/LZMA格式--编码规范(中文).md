# LZMA 编码规范（草案版本）

## 概述

- 作者：Igor Pavlov
- 日期：2015-06-14

本规范定义了 LZMA 压缩数据的格式以及 lzma 文件格式。

## 符号说明

我们使用 C++ 编程语言的语法。

在 C++ 代码中使用以下类型：

- `unsigned` - 无符号整型，至少 16 位
- `int` - 有符号整型，至少 16 位
- `UInt64` - 64 位无符号整型
- `UInt32` - 32 位无符号整型
- `UInt16` - 16 位无符号整型
- `Byte` - 8 位无符号整型
- `bool` - 布尔类型，取值为 `false` 或 `true`

---

## LZMA 文件格式

LZMA 文件包含原始的 LZMA 数据流以及相关的属性头。

此类文件使用 ".lzma" 扩展名。

### LZMA 文件格式布局

| 偏移量 | 大小 | 描述 |
|--------|------|------|
| 0      | 1    | LZMA 模型属性（lc、lp、pb）的编码形式 |
| 1      | 4    | 字典大小（32 位无符号整型，小端序） |
| 5      | 8    | 未压缩数据大小（64 位无符号整型，小端序） |
| 13     |      | 压缩数据（LZMA 流） |

### LZMA 属性

| 名称      | 范围           | 描述       | 中文       |
|-----------|----------------|------------|------------|
| lc        | [0, 8]         | "literal context" 的 bit 位数 | 字面量上下文 |
| lp        | [0, 4]         | "literal pos" 的 bit 位数 | 字面量位置 |
| pb        | [0, 4]         | "pos" 的 bit 位数 | 位置 |
| dictSize  | [0, 2^32 - 1]  | "dictionary size" | 字典大小 |

以下代码用于编码 LZMA 属性：

```cpp
void EncodeProperties(Byte * properties)
{
  properties[0] = (Byte)((pb * 5 + lp) * 9 + lc);
  Set_UInt32_LittleEndian(properties + 1, dictSize);
}
```

如果属性中的字典大小值小于 `(1 << 12)`，LZMA 解码器必须将字典大小变量设置为 `(1 << 12)`。

```cpp
#define LZMA_DIC_MIN (1 << 12)

unsigned lc, pb, lp;
UInt32 dictSize;
UInt32 dictSizeInProperties;

void DecodeProperties(const Byte *properties)
{
  unsigned d = properties[0];
  if (d >= (9 * 5 * 5)) {
    throw "Incorrect LZMA properties";
  }
  lc = d % 9;
  d /= 9;
  pb = d / 5;
  lp = d % 5;
  dictSizeInProperties = 0;
  for (int i = 0; i < 4; i++) {
    dictSizeInProperties |= (UInt32)properties[i + 1] << (8 * i);
  }
  dictSize = dictSizeInProperties;
  if (dictSize < LZMA_DIC_MIN)
    dictSize = LZMA_DIC_MIN;
}
```

如果 "未压缩大小" 字段的所有 64 bit 均为 1，则表示未压缩大小未知，并且数据流中存在 "结束标记"，用于指示解码结束点。

反之，如果 "未压缩大小" 字段的值不等于 `(2^64 - 1)`，则必须在解码指定数量的字节（未压缩大小）后完成 LZMA 流的解码。如果存在 "结束标记"，LZMA 解码器也必须读取该标记。

---

### 编码 LZMA 属性的新方案

如果 LZMA 压缩用于其他格式，建议使用新的改进方案来编码 LZMA 属性。该新方案已在 xz 格式中使用，该格式采用基于 LZMA 算法的 LZMA2 压缩算法。

LZMA2 中的字典大小仅用一个字节编码，并且 LZMA2 仅支持一组简化的字典大小：

```
`(2 << 11)`、`(3 << 11)`、
`(2 << 12)`、`(3 << 12)`、
...、
`(2 << 30)`、`(3 << 30)`、
`(2 << 31) - 1`
```

可以通过以下代码从编码值中提取字典大小：

```cpp
dictSize = (p == 40) ? 0xFFFFFFFF : (((UInt32)2 | ((p) & 1)) << ((p) / 2 + 11));
```

此外，LZMA2 对 "lc" 和 "lp" 属性有额外限制（`lc + lp <= 4`）：

```cpp
if (lc + lp > 4) {
  throw "Unsupported properties: (lc + lp) > 4";
}
```

这种限制对 LZMA 解码器有一些优势：

它减少了解码器分配数组的最大容量，并降低了初始化过程的复杂性，这对于保持大量小型 LZMA 流的高速解码非常重要。

建议在任何使用 LZMA 压缩的新格式中采用此限制（`lc + lp <= 4`）。注意，仅在某些罕见情况下，"lc" 和 "lp" 参数的组合（`lc + lp > 4`）才能显著提高压缩比。

在新方案中，LZMA 属性可以编码为两个字节：

| 偏移量 | 大小 | 描述 |
|--------|------|------|
| 0      | 1    | 使用 LZMA2 方案编码的字典大小 |
| 1      | 1    | LZMA 模型属性（lc、lp、pb）的编码形式 |

---

## 内存使用情况

LZMA 解码器的内存使用情况由以下部分组成：

1) 滑动窗口（从 4 KiB 到 4 GiB）。
2) 概率模型计数器数组（16-bit 变量的数组）。
3) 一些额外的状态变量（约 10 个 32 位整型变量）。

### 滑动窗口的内存使用情况

解码的时候，有两种主要场景：

1) 将完整流解码到一个内存缓冲区。

    如果我们将完整的 LZMA 流解码到内存中的一个输出缓冲区，解码器可以将该输出缓冲区用作滑动窗口。所以解码器不需要为滑动窗口分配额外的缓冲区。

2) 解码到外部存储。

    如果我们将 LZMA 流解码到外部存储，解码器必须为滑动窗口分配缓冲区，其大小必须大于或等于 LZMA 流属性中的字典大小值。

在本规范中，我们描述了用于解码到某些外部存储的代码。用于将完整的 LZMA 流解码到内存中的一个输出缓冲区的优化版本，可能需要对代码进行一些微小的更改。

### 概率模型计数器的内存使用情况

概率模型计数器数组的大小通过以下公式计算：

```c
size_of_prob_arrays = 1846 + 768 * (1 << (lp + lc))
```

每个概率模型计数器是一个 11-bit 的无符号整型。

如果使用 16-bit 整型变量（2 字节整型）存储这些计数器，则内存使用量可通过以下公式估算：

```c
RAM = 4 KiB + 1.5 KiB * (1 << (lp + lc))
```

例如，对于默认的 LZMA 参数（`lp = 0` 和 `lc = 3`），内存使用量为：

```c
RAM_lc3_lp0 = 4 KiB + 1.5 KiB * 8 = 16 KiB
```

最大内存使用量出现在 `lp = 4` 和 `lc = 8` 时：

```c
RAM_lc8_lp4 = 4 KiB + 1.5 KiB * 4096 = 6148 KiB
```

如果解码器采用 LZMA2 的限制条件（`lc + lp <= 4`），内存使用量不会超过：

```c
RAM_lc_lp_4 = 4 KiB + 1.5 KiB * 16 = 28 KiB
```

### 编码器的内存使用情况

LZMA 编码器有许多变体，其内存消耗各不相同。注意，LZMA 编码器的内存消耗不能小于相同流的 LZMA 解码器的内存消耗。

现代高效 LZMA 编码器的内存使用量可通过以下公式估算：

```c
Encoder_RAM_Usage = 4 MiB + 11 * dictionarySize
```

但某些编码模式需要较少的内存。

---

## LZMA 解码

LZMA 压缩算法使用基于 LZ77 的滑动窗口压缩和范围编码作为熵编码方法。

### 滑动窗口

LZMA 采用与 LZ77 算法类似的滑动窗口技术。

LZMA 数据流必须解码为由以下两种元素组成的序列：

- **字面量（LITERAL）**：8-bit 字符（1 个字节），解码器只需将该字面量（LITERAL）放入解压后的数据流中。

- **匹配（MATCH）**：由两个数字组成的匹配对（距离-长度），解码器从滑动窗口中复制指定距离和长度的字节序列。

**距离值限制**：

- 距离值不能超过字典大小（即滑动窗口大小）
- 距离值不能超过该匹配之前已解码的字节数

在本规范中，我们使用循环缓冲区来实现 LZMA 解码器的滑动窗口：

```cpp
class COutWindow
{
  Byte * Buf;         // 缓冲区指针
  UInt32 Pos;         // 当前位置
  UInt32 Size;        // 缓冲区大小
  bool IsFull;        // 缓冲区是否已满

public:
  unsigned TotalPos;    // 总位置计数
  COutStream OutStream; // 输出流

  COutWindow(): Buf(NULL) {}
  ~COutWindow() { delete[] Buf; }

  // 创建指定大小的滑动窗口
  void Create(UInt32 dictSize) {
    Buf = new Byte[dictSize];
    Pos = 0;
    Size = dictSize;
    IsFull = false;
    TotalPos = 0;
  }

  // 向窗口放入一个字节
  void PutByte(Byte b)
  {
    TotalPos++;
    Buf[Pos++] = b;
    if (Pos == Size) { // 到达缓冲区末尾时循环
      Pos = 0;
      IsFull = true;
    }
    OutStream.WriteByte(b); // 同时写入输出流
  }

  // 获取相对当前位置dist距离处的字节
  Byte GetByte(UInt32 dist) const
  {
    return Buf[dist <= Pos ? Pos - dist : Size - dist + Pos];
  }

  // 复制匹配内容
  void CopyMatch(UInt32 dist, unsigned len)
  {
    for (; len > 0; len--) {
      PutByte(GetByte(dist));
    }
  }

  // 检查距离是否有效
  bool CheckDistance(UInt32 dist) const
  {
    return ((dist <= Pos) || IsFull);
  }

  // 检查窗口是否为空
  bool IsEmpty() const
  {
    return ((Pos == 0) && !IsFull);
  }
};
```

其他实现方式也可以使用单个缓冲区来同时包含滑动窗口和整个解压后的数据流。

## 范围解码器（Range Decoder）

LZMA 算法使用范围编码（Range Encoding）作为其熵编码方法，范围编码类似于算术编码。跟算术编码不同的是，使用整数表示范围，概率值也使用的是整数。

LZMA 数据流本质上是一个采用大端编码的极大整数。LZMA 解码器通过范围解码器从这个大整数中提取二进制符号序列。

### Range Decoder 状态结构

- `Range` 和 `Code` 变量（32 位无符号整型）。
- `Corrupted` 标志，用于检测数据流中的损坏。

```cpp
struct CRangeDecoder
{
  UInt32 Range;          // 当前范围值
  UInt32 Code;           // 当前编码值
  InputStream *InStream; // 输入流指针

  bool Corrupted;        // 数据损坏标志
};
```

### 关于 Range 和 Code 变量的说明

1. 可以使用 64 位（有符号或无符号）整型替代 32 位无符号整型来存储 Range 和 Code 变量，但需要在某些操作后截取低 32 位值。

2. 若编程语言不支持 32 位无符号整型（如 Java），可改用 32 位有符号整型，但需要修改相关比较操作的代码实现。

### 数据损坏检测

范围解码器通过 Corrupted 标志位标识数据流异常：

- `Corrupted == false`：未检测到数据损坏
- `Corrupted == true`：检测到数据损坏

注：标准 LZMA 解码器会忽略 Corrupted 标志，即使检测到损坏仍会继续解码。为保证与标准解码器的输出兼容，其他实现也应遵循此行为。LZMA 编码器必须确保生成的压缩流不会导致解码器设置 Corrupted 标志。

### 初始化过程

范围解码器通过读取输入流的前 5 个字节进行初始化：

```cpp
bool CRangeDecoder::Init()
{
  Corrupted = false;
  Range = 0xFFFFFFFF;  // 初始范围设为最大值
  Code = 0;            // 初始编码值清零

  Byte b = InStream->ReadByte();  // 读取首字节

  // 读取后续 4 个字节构建 Code 值
  for (int i = 0; i < 4; i++) {
    Code = (Code << 8) | InStream->ReadByte();
  }

  // 校验首字节必须为 0 且 Code 不等于 Range
  if (b != 0 || Code == Range) {
    Corrupted = true;
  }

  return (b == 0);  // 返回首字节是否为0的校验结果
}
```

特别说明：

1. LZMA 编码器始终将压缩流的首字节中写入零，这种设计简化了编码器的范围编码实现。
2. 若解码器检测到首字节不为零，必须立即终止解码并报错。

### 数据完整性校验

当范围解码器完成最后一位数据的解码后，"Code" 变量的值必须等于 0。LZMA 解码器需要通过调用 IsFinishedOK() 函数进行验证：

```cpp
bool IsFinishedOK() const { return Code == 0; }
```

若数据流存在损坏，在 Finish() 函数中 "Code" 值极可能不为 0。因此 IsFinishedOK() 函数的这项检查为数据损坏检测提供了重要保障。

### 范围值规范化

每次位解码前，"Range" 变量的值不得小于 ((UInt32)1 << 24)。Normalize() 函数确保 "Range" 值维持在该范围内：

```cpp
#define kTopValue ((UInt32)1 << 24)

void CRangeDecoder::Normalize()
{
  if (Range < kTopValue) { // 当范围值低于阈值时
    Range <<= 8;           // 范围值左移 8 位
    Code = (Code << 8) | InStream->ReadByte();  // 同步更新编码值
  }
}
```

重要说明：

1. 若 "Code" 变量位宽超过 32 位，在 Normalize() 函数操作后需仅保留低 32 位值。

2. 对于完好的 LZMA 流，"Code" 值始终小于 "Range" 值。

3. 由于范围解码器会忽略某些类型的数据损坏，在部分损坏的压缩包中可能出现 "Code" 值大于或等于 "Range" 值的情况。

该机制通过动态调整解码范围来保证解码精度，同时利用终态校验为数据完整性提供了有效验证手段。规范化操作通过字节级再填充确保了解码过程的数值稳定性。

### 二进制符号处理

LZMA 算法仅使用两种类型的二进制符号进行范围编码：

1. **固定概率**：基于固定和相等概率的二进制符号（direct bits）。
2. **动态预测概率**：基于动态预测概率的二进制符号。

### Direct bits 解码函数

DecodeDirectBits() 函数用于解码 direct bits 序列：

```cpp
UInt32 CRangeDecoder::DecodeDirectBits(unsigned numBits)
{
  UInt32 res = 0; // 初始化结果变量
  do {
    Range >>= 1;   // 范围值减半
    Code -= Range; // 调整编码值

    // 计算符号位（Code 的符号位取反）
    UInt32 t = 0 - ((UInt32)Code >> 31);
    Code += Range & t; // 条件恢复 Code 值

    // 异常检测：当 Code 等于 Range 时标记数据损坏
    if (Code == Range)
      Corrupted = true;

    Normalize();  // 执行范围规范化
    res <<= 1;    // 结果左移腾出新位
    res += t + 1; // 存储解码得到的位值
  } while (--numBits); // 循环处理指定位数

  return res;
}
```

该函数通过迭代方式解码指定位数的 direct bits：

1. 每次迭代处理 1 个二进制位；
2. 采用算术运算动态调整 Range 和 Code 值；
3. 自动检测 Code == Range 的异常情况；
4. 通过 Normalize() 维持解码精度；
5. 最终返回拼接好的位序列。

注：`t` 变量的巧妙运用实现了符号位的快速判断与条件补偿，这是范围解码器的核心操作之一。

## 基于概率模型的位解码（Bit decoding）

概率模型的任务是估计二进制符号的概率，并将该信息提供给范围解码器（Range Decoder），更好的预测能够提供更好的压缩比。

概率值以 11-bit 无符号整型的形式表示二进制符号 `0` 或 `1` 的概率。

```cpp
#define kNumBitModelTotalBits 11

Mathematical probabilities can be presented with the following formulas:
     probability(symbol_0) = prob / 2048.
     probability(symbol_1) =  1 - Probability(symbol_0)
                           =  1 - prob / 2048
                           =  (2048 - prob) / 2048
```

这里 "prob" 变量是一个 11-bit 整型的概率计数器。

建议使用 16 位的无符号整型来存储这些 11-bit 的概率值：

```cpp
typedef UInt16 CProb;
```

每个概率值必须用值 `((1 << 11) / 2)` 来初始化，其表示二进制符号 `0` 和 `1` 的概率等于 `0.5` 的状态：

```cpp
#define PROB_INIT_VAL ((1 << kNumBitModelTotalBits) / 2)
```

`INIT_PROBS` 宏用于初始化 `CProb` 变量数组：

```cpp
#define INIT_PROBS(p) \
  { for (unsigned i = 0; i < sizeof(p) / sizeof(p[0]); i++) p[i] = PROB_INIT_VAL; }
```

DecodeBit() 函数解码一个 bit 。

LZMA 解码器提供指向 CProb 变量的指针，该变量包含关于符号 `0` 的估计概率，并且距离解码器（Range Decoder）在解码后更新 CProb 变量的概率。

范围解码器（Range Decoder）是如何在解码一个符号后更新估计概率的：

```cpp
#define kNumBitModelTotalBits 11
#define kNumMoveBits 5

unsigned CRangeDecoder::DecodeBit(CProb *prob)
{
  unsigned v = *prob;
  UInt32 bound = (Range >> kNumBitModelTotalBits) * v;
  unsigned symbol;
  if (Code < bound) {
    v += ((1 << kNumBitModelTotalBits) - v) >> kNumMoveBits;
    Range = bound;
    symbol = 0;
  } else {
    v -= v >> kNumMoveBits;
    Code -= bound;
    Range -= bound;
    symbol = 1;
  }
  *prob = (CProb)v;
  Normalize();
  return symbol;
}
```

## 位模型计数器的二叉树结构

LZMA 采用二叉树结构的位模型（bit model）变量来解码需要多个 bit 存储的符号。

该算法包含两种二叉树变体：

1. **常规方案**：从高位到低位解码 bits 序列
2. **反向方案**：从低位到高位解码 bits 序列

每个二叉树结构支持不同大小的解码符号（包含符号值的二进制序列的大小）。

如果解码符号的大小为 “NumBits” 位，则树结构使用包含 (2 << NumBits) 个 CProb 类型的计数器数组。

但编码器和解码器只使用了 ((2 << NumBits) - 1) 项，数组中的第一项（索引为 0 的项）保留未使用，使用未使用数组项的方案可以简化代码。

### 树结构特性

| 参数         | 说明                                                                 |
|--------------|----------------------------------------------------------------------|
| 符号位宽     | 由 `NumBits` 参数定义符号的二进制序列长度                            |
| 概率计数器数 | 使用 `(2 << NumBits)` 个 CProb 类型计数器组成的数组                  |
| 实际使用项   | 仅使用 `((2 << NumBits) - 1)` 个元素                                 |
| 特殊设计     | 数组索引 0 的位置保留未使用，此设计简化了编解码实现                  |

### 实现细节说明

1. **空间优化**：虽然数组大小为 `2 ^ (NumBits + 1)` ，但实际仅使用 `2 ^ (NumBits+1) - 1` 个有效项
2. **索引设计**：通过保留索引 0 的空位，使得子节点计算可通过 `m * 2` 和 `m * 2 + 1` 直接定位
3. **内存布局**：概率计数器数组按深度优先顺序排列，支持高效的缓存访问模式
4. **概率更新**：每个解码操作后自动更新对应节点的概率估计值，实现自适应编码

该二叉树结构通过分层概率建模，显著提升了长符号序列的解码效率，同时保持内存访问的局部性特征。

反向解码方案特别适用于需要低位优先处理的特定数据模式。

### 反向解码函数示例

```cpp
unsigned BitTreeReverseDecode(CProb * probs, unsigned numBits, CRangeDecoder * rc)
{
  unsigned m = 1;         // 初始节点指针
  unsigned symbol = 0;    // 符号值容器
  for (unsigned i = 0; i < numBits; i++) {
    unsigned bit = rc->DecodeBit(&probs[m]);  // 解码单个比特
    m <<= 1;              // 节点指针左移
    m += bit;             // 根据比特值选择子树
    symbol |= (bit << i); // 构建最终符号值
  }
  return symbol;
}

template <unsigned NumBits>
class CBitTreeDecoder
{
  CProb Probs[(unsigned)1 << NumBits];

public:
  void Init()
  {
    INIT_PROBS(Probs);
  }

  unsigned Decode(CRangeDecoder *rc)
  {
    unsigned m = 1;
    for (unsigned i = 0; i < NumBits; i++) {
      m = (m << 1) + rc->DecodeBit(&Probs[m]);
    }
    return m - ((unsigned)1 << NumBits);
  }

  unsigned ReverseDecode(CRangeDecoder *rc)
  {
    return BitTreeReverseDecode(Probs, NumBits, rc);
  }
};
```

## LZMA 的 LZ 部分

LZMA 的 LZ 部分详细描述了 **LITERALS(字面量)**  和 **MATCHES(匹配)** 的解码过程，本节重点解析字面量解码机制。

### 字面量解码（LITERALS）

#### 字面量概率表

LZMA 解码器使用了 (1 << (lc + lp)) 个包含 CProb 值的概率表，其中每个概率表包含 768 (0x300) 个 CProb 值。

其内存布局如下：

```cpp
  CProb * LitProbs; // 字面量概率表指针

  // 创建概率表（内存分配）
  void CreateLiterals() {
    // 概率表总数量 = 2 ^ (lc + lp)
    // 概率表单表大小 = 0x300（768个概率值）
    LitProbs = new CProb[(UInt32)0x300 << (lc + lp)];
  }

  // 初始化概率表（设为中间值）
  void InitLiterals() {
    UInt32 total = (UInt32)0x300 << (lc + lp);
    for (UInt32 i = 0; i < total; i++) {
      // 初始概率值 = 0x400，表示概率为 0.5
      LitProbs[i] = PROB_INIT_VAL;
    }
  }
```

它使用由上下文前一个字面量的高 lc 位和 outputStream 当前位置值的低 lp 位组合而成的复合键值 State，用于确定使用哪个概率表来解码当前的字面量。

##### 核心参数

| 参数 | 作用域 | 说明 |
|-----|-------|-----|
| `lc` | [0,8] | 字面量上下文位数|
| `lp` | [0,4] | 位置相关位数 |

##### 上下文选择机制

解码器通过以下方式确定当前使用的概率表：

1. **上下文继承**：取前一字面量的高 `lc` 位

```cpp
UInt32 ctxBits = prevByte >> (8 - lc)
```

2. **位置掩码**：取当前输出位置的低 `lp` 位

```cpp
UInt32 posBits = TotalPos & ((1 << lp) - 1)
```

3. **复合键值**：

```cpp
UInt32 litState = (posBits << lc) | ctxBits
```

#### 解码器状态

如果 (State > 7) ，字面量解码器也会使用 "matchByte" 来表示 OutputStream 当前位置之前 DISTANCE 个字节，这里的 DISTANCE 来自于最近解码的匹配对（DISTANCE-LENGTH pair）。

##### 状态机特性

| 状态值 | 解码模式 |
|-------|---------|
| state <7 | 纯概率解码 |
| state ≥7 | 混合参考字节预测 |

##### 预测优化机制

- **matchByte**：基于最近匹配距离（rep0）获取参考字节
- **位级预测**：将参考字节的对应位与当前解码位进行对比，动态调整解码路径
- **提前终止**：当实际解码位与参考位不一致时立即切换为常规解码模式

该设计通过结合历史匹配信息显著提升了重复模式数据的解码效率，同时保持对随机数据的高兼容性。

#### 字面量解码流程

下面的代码解码一个字面量，并且把它放入滑动窗口缓冲区中：

```cpp
void DecodeLiteral(unsigned state, UInt32 rep0) {
    // 获取前一个解码字节（距离1的位置）
    unsigned prevByte = OutWindow.IsEmpty() ? 0 : OutWindow.GetByte(1);

    // 计算上下文索引
    unsigned posMask = (1 << lp) - 1;
    unsigned litState = ((OutWindow.TotalPos & posMask) << lc) | (prevByte >> (8 - lc));
    CProb* probs = &LitProbs[0x300 * litState]; // 定位概率表

    unsigned symbol = 1; // 初始符号值

    // 状态机分支：当state>=7时启用匹配字节预测
    if (state >= 7) {
        unsigned matchByte = OutWindow.GetByte(rep0 + 1); // 获取参考字节
        do {
            unsigned matchBit = (matchByte >> 7) & 1;             // 提取参考位
            matchByte <<= 1;                                      // 左移准备下一位
            unsigned probIndex = ((1 + matchBit) << 8) + symbol;  // 计算概率索引
            unsigned bit = RangeDec.DecodeBit(&probs[probIndex]); // 解码预测位
            symbol = (symbol << 1) | bit;                         // 构建当前符号

            if (matchBit != bit) break; // 预测失败时退出循环
        } while (symbol < 0x100);       // 循环直到完成字节解码
    }

    // 常规解码模式
    while (symbol < 0x100) {
        unsigned bit = RangeDec.DecodeBit(&probs[symbol]); // 直接解码
        symbol = (symbol << 1) | bit;
    }

    OutWindow.PutByte((Byte)(symbol - 0x100)); // 写入最终字节
}
```

## LZMA 解码模式

LZMA 流有两种类型：

1) 带有 "流结束标记" 的流。
2) 不带有 "流结束标记" 的流。

LZMA 解码器支持三种解码模式：

1) 未定义解压大小，解码器在收到 "流结束标记" 后停止。
2) 解压大小已定义，流可能包含或不包含 "流结束标记"。
3) 解压大小已定义，且流必须包含 "流结束标记"。

### 主解码循环

LZMA 解码器的主循环包括以下步骤：

1) 检查流结束条件。
2) 解码匹配或字面量类型。
3) 如果是字面量，解码并放入滑动窗口。
4) 如果是匹配，解码长度和距离，并复制匹配的字节序列。

LZMA 使用三种匹配类型：

1) **简单匹配**：使用位模型编码的距离值。
2) **重复匹配**：使用距离历史表中的距离。
3) **短重复匹配**：单字节长度的匹配，使用距离历史表中的最新距离。

解码器维护最近使用的 4 个匹配距离的历史记录，并通过状态变量和概率模型选择匹配或字面量类型。

## 注意事项

本规范未描述支持部分解码的解码器实现变体。此类实现可能需要对 "流结束" 条件检查代码进行修改，并可能使用额外的状态码。

本规范使用 C++ 代码和模板简化描述。优化的 LZMA 解码器不需要模板，可以使用两个 CProb 变量数组：

1) 动态分配的字面量解码器数组。
2) 包含所有其他 CProb 变量的公共数组。

## 参考文献

<a id="referen1"></a>

1. G. N. N. Martin, Range encoding: an algorithm for removing redundancy from a digitized message, Video & Data Recording Conference, Southampton, UK, July 24-27, 1979.
