好的，这是整理成 Markdown 格式并移除 `[source: ##]` 标记的 LZMA 规范中文翻译：

# LZMA 规范 (草稿版本)

----------------------------------

## 概述

* 作者：Igor Pavlov
* 日期：2015-06-14

本规范定义了 LZMA 压缩数据的格式和 lzma 文件格式。

## 标记法

--------

使用 C++ 编程语言的语法。

在 C++ 代码中使用以下类型：

* `unsigned` - 无符号整数，至少 16 位
* `int`      - 有符号整数，至少 16 位
* `UInt64`   - 64 位无符号整数
* `UInt32`   - 32 位无符号整数
* `UInt16`   - 16 位无符号整数
* `Byte`     - 8 位无符号整数
* `bool`     - 布尔类型，有两个可能的值：`false`、`true`

## lzma 文件格式

================

lzma 文件包含原始 LZMA 流和带有相关属性的头部。

该格式的文件使用 ".lzma" 扩展名。

lzma 文件格式布局：

| 偏移量 | 大小 | 描述                                                     |
| :----- | :--- | :------------------------------------------------------- |
| 0      | 1    | LZMA 模型属性 (`lc`, `lp`, `pb`) 的编码形式                  |
| 1      | 4    | 字典大小 (32 位无符号整数，小端序)                           |
| 5      | 8    | 未压缩大小 (64 位无符号整数，小端序)                         |
| 13     |      | 压缩数据 (LZMA 流)                                         |

LZMA 属性：

| 名称     | 范围          | 描述                                           |
| :------- | :------------ | :--------------------------------------------- |
| `lc`     | [0, 8]        | "字面上下文 (literal context)" 比特的数量        |
| `lp`     | [0, 4]        | "字面位置 (literal pos)" 比特的数量            |
| `pb`     | [0, 4]        | "位置 (pos)" 比特的数量                        |
| `dictSize` | [0, 2^32 - 1] | 字典大小                                       |

以下代码对 LZMA 属性进行编码：

```cpp
void EncodeProperties(Byte *properties)
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
      throw "Incorrect LZMA properties"; // 抛出“不正确的 LZMA 属性”异常
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
    if (dictSize < LZMA_DIC_MIN) {
      dictSize = LZMA_DIC_MIN;
    }
  }
```

如果 "未压缩大小" 字段包含全为 1 的 64 位，则表示未压缩大小未知，并且流中存在指示解码结束点的 "结束标记"。相反，如果 "未压缩大小" 字段的值不等于 `((2^64) - 1)`，则 LZMA 流解码必须在解码了指定数量的字节（未压缩大小）后完成。

并且如果存在 "结束标记"，LZMA 解码器也必须读取该标记。

## 编码 LZMA 属性的新方案

----------------------------------------

如果 LZMA 压缩用于其他某种格式，建议使用一种新的改进方案来编码 LZMA 属性。这个新方案已在 `xz` 格式中使用，该格式使用了 `LZMA2` 压缩算法。

`LZMA2` 是一种基于 LZMA 算法的新压缩算法。

在 `LZMA2` 中，字典大小仅用一个字节编码，并且 `LZMA2` 仅支持字典大小的缩减集：

* `(2 << 11), (3 << 11),`
* `(2 << 12), (3 << 12),`
* ...
* `(2 << 30), (3 << 30),`
* `(2 << 31) - 1`

可以使用以下代码从编码值中提取字典大小：

```cpp
dictSize = (p == 40) ? 0xFFFFFFFF
        : (((UInt32)2 | ((p) & 1)) << ((p) / 2 + 11));
```

此外，在 `LZMA2` 中对 `"lc"` 和 `"lp"` 属性的值还有额外的限制 `(lc + lp <= 4)`：

```cpp
// 抛出 "不支持的属性：(lc + lp) > 4" 的异常
if (lc + lp > 4) {
  throw "Unsupported properties: (lc + lp) > 4";
}
```

对于 LZMA 解码器来说，这样的 `(lc + lp)` 值限制有一些优点，它减少了解码器分配的表的最大大小。并且它降低了初始化过程的复杂性，这对于保持解码大量小型 LZMA 流的高速度可能很重要。

建议任何使用 LZMA 压缩的新格式都使用该限制 `(lc + lp <= 4)`。

请注意，`(lc + lp > 4)` 的 `"lc"` 和 `"lp"` 参数组合仅在某些罕见情况下才能显著提高压缩率。

在新方案中，LZMA 属性可以编码为两个字节：

| 偏移量 | 大小 | 描述                                  |
| :----- | :--- | :------------------------------------ |
| 0      | 1    | 使用 LZMA2 方案编码的字典大小           |
| 1      | 1    | LZMA 模型属性 (`lc`, `lp`, `pb`) 的编码形式 |

## RAM 使用

=============

LZMA 解码器的 RAM 使用由以下部分决定：

1. 滑动窗口（从 4 KiB 到 4 GiB）。
2. 概率模型计数器数组（16 位变量的数组）。
3. 一些额外的状态变量（大约 10 个 32 位整数变量）。

### 滑动窗口的 RAM 使用

--------------------------------

解码有两种主要场景：

1. **将完整流解码到一个 RAM 缓冲区中。**

    如果我们将完整的 LZMA 流解码到 RAM 中的一个输出缓冲区，解码器可以将该输出缓冲区用作滑动窗口。因此，解码器不需要为滑动窗口分配额外的缓冲区。

2. **解码到某个外部存储。**

  * 如果我们将 LZMA 流解码到外部存储，解码器必须为滑动窗口分配缓冲区。
  * 该缓冲区的大小必须等于或大于 LZMA 流属性中的字典大小值。
  * 在本规范中，我们描述了用于解码到某个外部存储的代码。
  * 用于将完整流解码到一个输出 RAM 缓冲区的优化版本代码可能需要在代码中进行一些细微更改。

### 概率模型计数器的 RAM 使用

------------------------------------------------

概率模型计数器数组的大小使用以下公式计算：

```cpp
size_of_prob_arrays = 1846 + 768 * (1 << (lp + lc))
```

每个概率模型计数器是 11 位无符号整数。

如果我们对这些概率模型计数器使用 16 位整数变量（2 字节整数），则概率模型计数器数组所需的 RAM 使用量可以通过以下公式估算：

```cpp
RAM = 4 KiB + 1.5 KiB * (1 << (lp + lc))
```

例如，对于默认的 LZMA 参数（`lp = 0` 和 `lc = 3`），RAM 使用量为：

```cpp
RAM_lc3_lp0 = 4 KiB + 1.5 KiB * 8 = 16 KiB
```

解码具有 `lp = 4` 和 `lc = 8` 的流需要最大的 RAM 状态使用量：

```cpp
RAM_lc8_lp4 = 4 KiB + 1.5 KiB * 4096 = 6148 KiB
```

如果解码器使用 `LZMA2` 的受限属性条件 `(lc + lp <= 4)`，则 RAM 使用量将不大于

```cpp
RAM_lc_lp_4 = 4 KiB + 1.5 KiB * 16 = 28 KiB
```

### 编码器的 RAM 使用

-------------------------

LZMA 编码代码有许多变体。这些变体具有不同的内存消耗值。请注意，对于同一流，LZMA 编码器的内存消耗不能小于 LZMA 解码器的内存消耗。

现代高效 LZMA 编码器实现所需的 RAM 使用量可以通过以下公式估算：

`Encoder_RAM_Usage = 4 MiB + 11 * dictionarySize`.
但也有一些编码器模式需要更少的内存。

## LZMA 解码

=============

LZMA 压缩算法使用基于 LZ 的滑动窗口压缩，并使用范围编码 (Range Encoding) 作为熵编码方法。

### 滑动窗口

--------------

LZMA 使用类似于 LZ77 算法的滑动窗口压缩。

LZMA 流必须解码为由匹配 (MATCHES) 和字面值 (LITERALS) 组成的序列：

* **字面值 (LITERAL)** 是一个 8 位字符（一个字节）。解码器只需将该字面值放入未压缩流中。

* **匹配 (MATCH)** 是一对数字（距离-长度对 (`DISTANCE`-`LENGTH` pair)）。解码器从解压缩流中当前位置向前 `DISTANCE` 个字符处精确地取一个字节，并将其放入解压缩流中，解码器必须重复此操作 `LENGTH` 次。

`DISTANCE` 不能大于字典大小，并且 `DISTANCE` 不能大于在该匹配之前已解码的未压缩流中的字节数。

在本规范中，我们使用循环缓冲区来实现 LZMA 解码器的滑动窗口：

```cpp
class COutWindow
{
  Byte *Buf;
  UInt32 Pos;
  UInt32 Size;
  bool IsFull;

public:
  unsigned TotalPos;
  COutStream OutStream;

  COutWindow(): Buf(NULL) {}
  ~COutWindow() { delete []Buf; }

  void Create(UInt32 dictSize)
  {
    Buf = new Byte[dictSize];
    Pos = 0;
    Size = dictSize;
    IsFull = false;
    TotalPos = 0;
  }

  void PutByte(Byte b)
  {
    TotalPos++;
    Buf[Pos++] = b;
    if (Pos == Size)
    {
      Pos = 0;
      IsFull = true;
    }
    OutStream.WriteByte(b);
  }

  Byte GetByte(UInt32 dist) const
  {
    return Buf[dist <= Pos ?
    Pos - dist : Size - dist + Pos];
  }

  void CopyMatch(UInt32 dist, unsigned len)
  {
    for (; len > 0; len--)
      PutByte(GetByte(dist));
  }

  bool CheckDistance(UInt32 dist) const
  {
    return dist <= Pos || IsFull;
  }

  bool IsEmpty() const
  {
    return Pos == 0 && !IsFull;
  }
};
```

在另一种实现中，可以使用一个缓冲区，该缓冲区包含滑动窗口和解压缩后的整个数据流。

### 范围解码器 (Range Decoder)

-------------

LZMA 算法使用范围编码 (Range Encoding) (1) 作为熵编码方法。

LZMA 流仅包含一个大端序编码的非常大的数字。LZMA 解码器使用范围解码器从该大数中提取二进制符号序列。

范围解码器的状态：

```cpp
struct CRangeDecoder
{
  UInt32 Range;
  UInt32 Code;
  InputStream *InStream;

  bool Corrupted;
};
```

关于 `Range` 和 `Code` 变量的 `UInt32` 类型的说明：

* 可以使用 64 位（无符号或有符号）整数类型代替 32 位无符号整数来表示 `Range` 和 `Code` 变量，但必须使用一些额外的代码在某些操作后将值截断为低 32 位。
* 如果编程语言不支持 32 位无符号整数类型（如 JAVA 语言），可以使用 32 位有符号整数，但必须更改一些代码。
* 例如，需要更改本规范中使用 `UInt32` 变量比较操作的代码。

范围解码器可能处于某些状态，这些状态可被视为 LZMA 流中的 "损坏 (Corruption)"。

范围解码器使用变量 `Corrupted`：

* (`Corrupted == false`)，如果范围解码器未检测到任何损坏。
* (`Corrupted == true`)，如果范围解码器检测到某些损坏。

参考 LZMA 解码器会忽略 `Corrupted` 变量的值。因此，即使在范围解码器中可以检测到损坏，它也会继续解码流。为了提供与参考 LZMA 解码器输出的完全兼容性，其他 LZMA 解码器实现也必须忽略 `Corrupted` 变量的值。

LZMA 编码器必须仅创建不会导致范围解码器进入 `Corrupted` 变量设置为 true 的状态的 LZMA 流。

范围解码器从输入流读取前 5 个字节以初始化状态：

```cpp
bool CRangeDecoder::Init()
{
  Corrupted = false;
  Range = 0xFFFFFFFF;
  Code = 0;

  Byte b = InStream->ReadByte();
  for (int i = 0; i < 4; i++)
    Code = (Code << 8) | InStream->ReadByte();
  if (b != 0 || Code == Range)
    Corrupted = true;
  return b == 0;
}
```

LZMA 编码器总是在压缩流的初始字节中写入零。该方案允许简化 LZMA 编码器中范围编码器的代码。如果初始字节不等于零，LZMA 解码器必须停止解码并报告错误。

在范围解码器解码完数据的最后一位后，`Code` 变量的值必须等于 0。LZMA 解码器必须通过调用 `IsFinishedOK()` 函数来检查它：

```cpp
  bool IsFinishedOK() const { return Code == 0; }
```

如果数据流中存在损坏，则在 `Finish()` 函数中 `Code` 值不等于 0 的概率很大。因此，`IsFinishedOK()` 函数中的检查为损坏检测提供了非常好的功能。

每次位解码之前的 `Range` 变量的值不能小于 `((UInt32)1 << 24)`。

`Normalize()` 函数将 `Range` 值保持在所述范围内。

```cpp
#define kTopValue ((UInt32)1 << 24)

void CRangeDecoder::Normalize()
{
  if (Range < kTopValue)
  {
    Range <<= 8;
    Code = (Code << 8) | InStream->ReadByte();
  }
}
```

注意：如果 `Code` 变量的大小大于 32 位，则需要在 `Normalize()` 函数更改后仅保留 `Code` 变量的低 32 位。如果 LZMA 流未损坏，`Code` 变量的值始终小于 `Range` 变量的值。但是范围解码器会忽略某些类型的损坏，因此对于某些 "损坏的" 存档，`Code` 变量的值可能等于或大于 `Range` 变量的值。

LZMA 仅将范围编码用于两种类型的二进制符号：

1)  具有固定且相等概率的二进制符号（直接位）
2)  具有预测概率的二进制符号

`DecodeDirectBits()` 函数解码直接位序列：

```cpp
UInt32 CRangeDecoder::DecodeDirectBits(unsigned numBits)
{
  UInt32 res = 0;
  do
  {
    Range >>= 1;
    Code -= Range;
    UInt32 t = 0 - ((UInt32)Code >> 31);
    Code += Range & t;

    if (Code == Range)
      Corrupted = true;

    Normalize();
    res <<= 1;
    res += t + 1;
  }
  while (--numBits);
  return res;
}
```

### 使用概率模型进行位解码

---------------------------------------

位概率模型的任务是估计二进制符号的概率，然后它向范围解码器提供该信息，更好的预测可提供更好的压缩率。

位概率模型使用先前解码符号的统计数据。该估计概率表示为一个 11 位无符号整数值，代表符号 "0" 的概率。

```cpp
#define kNumBitModelTotalBits 11
```

数学概率可以用以下公式表示：

```cpp
probability(symbol_0) = prob / 2048.
probability(symbol_1) = 1 - Probability(symbol_0)
                      = 1 - prob / 2048
                      = (2048 - prob) / 2048
```

其中 `prob` 变量包含 11 位整数概率计数器。

建议使用 16 位无符号整数类型 (`UInt16`) 来存储这些 11 位概率值：

```cpp
typedef UInt16 CProb;
```

每个概率值必须初始化为 `((1 << 11) / 2)`，该值表示符号 0 和 1 的概率等于 0.5 的状态：

```cpp
#define PROB_INIT_VAL ((1 << kNumBitModelTotalBits) / 2)
```

`INIT_PROBS` 宏用于初始化 `CProb` 变量数组：

```cpp
#define INIT_PROBS(p) \
  { for (unsigned i = 0; i < sizeof(p) / sizeof(p[0]); i++) p[i] = PROB_INIT_VAL; }
```

`DecodeBit()` 函数解码一位。
LZMA 解码器提供指向 `CProb` 变量的指针，该变量包含有关符号 0 的估计概率的信息，范围解码器在解码后更新该 `CProb` 变量。范围解码器增加已解码符号的估计概率：

```cpp
#define kNumMoveBits 5

unsigned CRangeDecoder::DecodeBit(CProb *prob)
{
  unsigned v = *prob;
  UInt32 bound = (Range >> kNumBitModelTotalBits) * v;
  unsigned symbol;
  if (Code < bound)
  {
    v += ((1 << kNumBitModelTotalBits) - v) >> kNumMoveBits;
    Range = bound;
    symbol = 0;
  }
  else
  {
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

### 位模型计数器的二叉树

-------------------------------------

LZMA 使用位模型变量树来解码需要多个位来存储的符号。

LZMA 中有两种这样的树：

1) 从高位到低位解码位的树（正常方案）。
2) 从低位到高位解码位的树（反向方案）。

每个二叉树结构支持不同大小的解码符号（包含符号值的二进制序列的大小）。如果解码符号的大小为 `NumBits` 位，则树结构使用 `(2 << NumBits)` 个 `CProb` 类型计数器的数组，但编码器和解码器仅使用 `((2 << NumBits) - 1)` 个项。数组中的第一项（索引等于 0 的项）未使用，这种使用未使用数组项的方案允许简化代码。

```cpp
unsigned BitTreeReverseDecode(CProb *probs, unsigned numBits, CRangeDecoder *rc)
{
  unsigned m = 1;
  unsigned symbol = 0;
  for (unsigned i = 0; i < numBits; i++)
  {
    unsigned bit = rc->DecodeBit(&probs[m]);
    m <<= 1;
    m += bit;
    symbol |= (bit << i);
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
    for (unsigned i = 0; i < NumBits; i++)
      m = (m << 1) + rc->DecodeBit(&Probs[m]);
    return m - ((unsigned)1 << NumBits);
  }

  unsigned ReverseDecode(CRangeDecoder *rc)
  {
    return BitTreeReverseDecode(Probs, NumBits, rc);
  }
};
```

## LZMA 的 LZ 部分

---------------

LZMA 的 LZ 部分描述了有关解码匹配 (MATCHES) 和字面值 (LITERALS) 的详细信息。

### 字面值解码

--------------------

LZMA 解码器使用 `(1 << (lc + lp))` 个包含 `CProb` 值的表，其中每个表包含 0x300 个 `CProb` 值：

```cpp
  CProb *LitProbs;
  void CreateLiterals()
  {
    LitProbs = new CProb[(UInt32)0x300 << (lc + lp)];
  }

  void InitLiterals()
  {
    UInt32 num = (UInt32)0x300 << (lc + lp);
    for (UInt32 i = 0; i < num; i++)
      LitProbs[i] = PROB_INIT_VAL;
  }
```

为了选择用于解码的表，它使用由前一个字面值的 (`lc`) 个高位和表示输出流中当前位置的值的 (`lp`) 个低位组成的上下文。

如果 `(State > 7)`，字面值解码器还使用 `matchByte`，它表示输出流中距离当前位置 `DISTANCE` 个字节的位置处的字节，其中 `DISTANCE` 是最近解码的匹配的距离-长度对中的距离。

以下代码解码一个字面值并将其放入滑动窗口缓冲区：

```cpp
  void DecodeLiteral(unsigned state, UInt32 rep0)
  {
    unsigned prevByte = 0;
    if (!OutWindow.IsEmpty()) {
      prevByte = OutWindow.GetByte(1);
    }

    unsigned symbol = 1;
    unsigned litState = ((OutWindow.TotalPos & ((1 << lp) - 1)) << lc) + (prevByte >> (8 - lc));
    CProb *probs = &LitProbs[(UInt32)0x300 * litState];

    if (state >= 7) {
      unsigned matchByte = OutWindow.GetByte(rep0 + 1);
      do {
        unsigned matchBit = (matchByte >> 7) & 1;
        matchByte <<= 1;
        unsigned bit = RangeDec.DecodeBit(&probs[((1 + matchBit) << 8) + symbol]);
        symbol = (symbol << 1) | bit;
        if (matchBit != bit)
          break;
      } while (symbol < 0x100);
    }
    while (symbol < 0x100) {
      symbol = (symbol << 1) | RangeDec.DecodeBit(&probs[symbol]);
    }
    OutWindow.PutByte((Byte)(symbol - 0x100));
  }
```

### 匹配长度解码

-------------------------

匹配长度解码器返回匹配的规范化（从零开始的值）长度。

可以使用以下代码将该值转换为匹配的实际长度：

```cpp
#define kMatchMinLen 2
matchLen = len + kMatchMinLen;
```

匹配长度解码器可以返回从 0 到 271 的值。相应的实际匹配长度值可以在 2 到 273 的范围内。

匹配长度编码使用以下方案：

| 二进制编码序列 | 二叉树结构         | 从零开始的匹配长度（二进制 + 十进制） |
| :------------- | :----------------- | :------------------------------------ |
| `0 xxx`        | `LowCoder[posState]` | `xxx`                               |
| `1 0 yyy`      | `MidCoder[posState]` | `yyy + 8`                           |
| `1 1 zzzzzzzz` | `HighCoder`        | `zzzzzzzz + 16`                       |

LZMA 使用位模型变量 `Choice` 来解码第一个选择位。

* 如果第一个选择位等于 0，解码器使用二叉树 `LowCoder[posState]` 来解码 3 位从零开始的匹配长度 (`xxx`)。
* 如果第一个选择位等于 1，解码器使用位模型变量 `Choice2` 来解码第二个选择位。
* 如果第二个选择位等于 0，解码器使用二叉树 `MidCoder[posState]` 来解码 3 位 `"yyy"` 值，并且从零开始的匹配长度等于 (`yyy + 8`)。
* 如果第二个选择位等于 1，解码器使用二叉树 `HighCoder` 来解码 8 位 `"zzzzzzzz"` 值，并且从零开始的匹配长度等于 (`zzzzzzzz + 16`)。

LZMA 使用 `posState` 值作为上下文来从 `LowCoder` 和 `MidCoder` 二叉树数组中选择二叉树：

```cpp
  unsigned posState = OutWindow.TotalPos & ((1 << pb) - 1);
```

长度解码器的完整代码：

```cpp
class CLenDecoder
{
  CProb Choice;
  CProb Choice2;
  CBitTreeDecoder<3> LowCoder[1 << kNumPosBitsMax];
  CBitTreeDecoder<3> MidCoder[1 << kNumPosBitsMax];
  CBitTreeDecoder<8> HighCoder;

public:

  void Init()
  {
    Choice = PROB_INIT_VAL;
    Choice2 = PROB_INIT_VAL;
    HighCoder.Init();
    for (unsigned i = 0; i < (1 << kNumPosBitsMax); i++)
    {
      LowCoder[i].Init();
      MidCoder[i].Init();
    }
  }

  unsigned Decode(CRangeDecoder *rc, unsigned posState)
  {
    if (rc->DecodeBit(&Choice) == 0)
      return LowCoder[posState].Decode(rc);
    if (rc->DecodeBit(&Choice2) == 0)
      return 8 + MidCoder[posState].Decode(rc);
    return 16 + HighCoder.Decode(rc);
  }
};
```

LZMA 解码器使用 `CLenDecoder` 类的两个实例。第一个实例用于 "简单匹配 (Simple Match)" 类型的匹配，第二个实例用于 "重复匹配 (Rep Match)" 类型的匹配：

```cpp
CLenDecoder LenDecoder;
CLenDecoder RepLenDecoder;
```

### 匹配距离解码

---------------------------

LZMA 支持最大 4 GiB 减 1 的字典大小，距离解码器解码的匹配距离值可以从 1 到 2^32。但是等于 2^32 的距离值用于指示 "流结束 (End of stream)" 标记。因此，用于 LZ 窗口匹配的实际最大匹配距离是 (2^32 - 1)。

LZMA 使用规范化的匹配长度（从零开始的长度）来计算上下文状态 `lenState` 以解码距离值：

```cpp
#define kNumLenToPosStates 4

unsigned lenState = len;
if (lenState > kNumLenToPosStates - 1) {
  lenState = kNumLenToPosStates - 1;
}
```

距离解码器返回 `dist` 值，该值是匹配距离的从零开始的值。
可以使用以下代码计算实际的匹配距离：

```cpp
matchDistance = dist + 1;
```

距离解码器的状态和初始化代码：

```cpp
  #define kEndPosModelIndex 14
  #define kNumFullDistances (1 << (kEndPosModelIndex >> 1))
  #define kNumAlignBits 4

  CBitTreeDecoder<6> PosSlotDecoder[kNumLenToPosStates];
  CProb PosDecoders[1 + kNumFullDistances - kEndPosModelIndex];
  CBitTreeDecoder<kNumAlignBits> AlignDecoder;

  void InitDist()
  {
    for (unsigned i = 0; i < kNumLenToPosStates; i++)
      PosSlotDecoder[i].Init();
    AlignDecoder.Init();
    INIT_PROBS(PosDecoders);
  }
```

在第一阶段，距离解码器使用来自 `PosSlotDecoder` 数组的位树解码器解码 6 位 `posSlot` 值。可以得到 2^6=64 个不同的 `posSlot` 值。

```cpp
  unsigned posSlot = PosSlotDecoder[lenState].Decode(&RangeDec);
```

距离值的编码方案如下表所示：

| posSlot (十进制) | 从零开始的距离 (二进制)              |
| :--------------- | :----------------------------------- |
| 0                | `0`                                  |
| 1                | `1`                                  |
| 2                | `10`                                 |
| 3                | `11`                                 |
| 4                | `10 x`                               |
| 5                | `11 x`                               |
| 6                | `10 xx`                              |
| 7                | `11 xx`                              |
| 8                | `10 xxx`                             |
| 9                | `11 xxx`                             |
| 10               | `10 xxxx`                            |
| 11               | `11 xxxx`                            |
| 12               | `10 xxxxx`                           |
| 13               | `11 xxxxx`                           |
| 14               | `10 yy zzzz`                         |
| 15               | `11 yy zzzz`                         |
| 16               | `10 yyy zzzz`                        |
| 17               | `11 yyy zzzz`                        |
| ...              | ...                                  |
| 62               | `10 yyyyyyyyyyyyyyyyyyyyyyyyyy zzzz` |
| 63               | `11 yyyyyyyyyyyyyyyyyyyyyyyyyy zzzz` |

其中：

* `"x ... x"` 表示使用二叉树和 "反向 (Reverse)" 方案编码的二进制符号序列。它为从 4 到 13 的每个 `posSlot` 使用单独的二叉树。
* `"y"` 表示使用范围编码器编码的直接位。
* `"zzzz"` 表示使用具有 "反向 (Reverse)" 方案的二叉树编码的四个二进制符号序列，其中一个公共二叉树 `"AlignDecoder"` 用于所有 `posSlot` 值。

策略：

- 如果 `(posSlot < 4)`，`dist` 值等于 `posSlot` 值。
- 如果 `(posSlot >= 4)`，解码器使用 `posSlot` 值来计算 `dist` 值的高位和低位的数量。
- 如果 `(4 <= posSlot < kEndPosModelIndex)`，解码器使用位树解码器（每个 `posSlot` 值一个单独的位树解码器）和 "反向 (Reverse)" 方案。在此实现中，我们使用一个 `CProb` 数组 `PosDecoders`，它包含所有这些位解码器的所有 `CProb` 变量。
- 如果 `(posSlot >= kEndPosModelIndex)`，则中间位被解码为来自 `RangeDecoder` 的直接位，低 4 位使用具有 "反向 (Reverse)" 方案的位树解码器 `AlignDecoder` 进行解码。

解码从零开始的匹配距离的代码：

```cpp
  unsigned DecodeDistance(unsigned len)
  {
    unsigned lenState = len;
    if (lenState > kNumLenToPosStates - 1)
      lenState = kNumLenToPosStates - 1;

    unsigned posSlot = PosSlotDecoder[lenState].Decode(&RangeDec);
    if (posSlot < 4)
      return posSlot;

    unsigned numDirectBits = (unsigned)((posSlot >> 1) - 1);
    UInt32 dist = ((2 | (posSlot & 1)) << numDirectBits);
    if (posSlot < kEndPosModelIndex)
      dist += BitTreeReverseDecode(PosDecoders + dist - posSlot, numDirectBits, &RangeDec);
    else
    {
      dist += RangeDec.DecodeDirectBits(numDirectBits - kNumAlignBits) << kNumAlignBits;
      dist += AlignDecoder.ReverseDecode(&RangeDec);
    }
    return dist;
  }
```

## LZMA 解码模式

-------------------

LZMA 流有 2 种类型：

1. 带有 "流结束 (End of stream)" 标记的流。
2. 不带 "流结束 (End of stream)" 标记的流。

LZMA 解码器支持 3 种解码模式：

1. **解包大小未定义。** LZMA 解码器在获取到 "流结束 (End of stream)" 标记后停止解码。

    该情况的输入变量：

    * `markerIsMandatory = true`
    * `unpackSizeDefined = false`
    * `unpackSize` 包含任意值

2. **解包大小已定义，并且 LZMA 解码器支持两种变体**，即流可以包含 "流结束 (End of stream)" 标记，或者流在没有 "流结束 (End of stream)" 标记的情况下结束。LZMA 解码器必须检测这些情况中的任何一种。

    该情况的输入变量：

    * `markerIsMandatory = false`
    * `unpackSizeDefined = true`
    * `unpackSize` 包含解包大小

3. **解包大小已定义，并且 LZMA 流必须包含 "流结束 (End of stream)" 标记**

    该情况的输入变量：

    * `markerIsMandatory = true`
    * `unpackSizeDefined = true`
    * `unpackSize` 包含解包大小

## 解码器的主循环

------------------------

LZMA 解码器的主循环：

1. 初始化 LZMA 状态。

2. **循环开始**

    * 检查 "流结束" 条件。
    * 解码 匹配 (MATCH) / 字面值 (LITERAL) 的类型。
    * 如果是字面值 (LITERAL)，解码字面值并将其放入窗口。
    * 如果是匹配 (MATCH)，解码匹配的长度和匹配距离。
    * 检查错误条件，检查流结束条件，并将匹配字节序列从滑动窗口复制到窗口中的当前位置。
    * 转到循环开始

3. **循环结束**

LZMA 解码器的参考实现使用 `unpackSize` 变量来保存输出流中剩余的字节数。因此，它在每次解码字面值 (LITERAL) 或匹配 (MATCH) 后减少 `unpackSize` 值。

以下代码包含循环开始时的 "流结束" 条件检查：

```cpp
  if (unpackSizeDefined && unpackSize == 0 && !markerIsMandatory) {
    if (RangeDec.IsFinishedOK()) {
      return LZMA_RES_FINISHED_WITHOUT_MARKER; // 返回：无标记完成
    }
  }
```

LZMA 使用三种类型的匹配：

1. **"简单匹配 (Simple Match)"** - 距离值使用位模型编码的匹配。
2. **"重复匹配 (Rep Match)"** - 使用距离历史表中的距离的匹配。
3. **"短重复匹配 (Short Rep Match)"** - 单字节长度的匹配，使用距离历史表中的最新距离。

LZMA 解码器保存解码器使用的最近 4 个匹配距离的历史记录。

这组 4 个变量包含从零开始的匹配距离，这些变量用零值初始化：

```cpp
  UInt32 rep0 = 0, rep1 = 0, rep2 = 0, rep3 = 0;
```

LZMA 解码器使用二进制模型变量来选择匹配 (MATCH) 或字面值 (LITERAL) 的类型：

```cpp
#define kNumStates 12
#define kNumPosBitsMax 4

CProb IsMatch[kNumStates << kNumPosBitsMax];
CProb IsRep[kNumStates];
CProb IsRepG0[kNumStates];
CProb IsRepG1[kNumStates];
CProb IsRepG2[kNumStates];
CProb IsRep0Long[kNumStates << kNumPosBitsMax];
```

解码器使用 `state` 变量值来从 `IsRep`、`IsRepG0`、`IsRepG1` 和 `IsRepG2` 数组中选择确切的变量。

`state` 变量可以取 0 到 11 的值。`state` 变量的初始值为零：

```cpp
unsigned state = 0;
```

`state` 变量在每次字面值 (LITERAL) 或匹配 (MATCH) 后使用以下函数之一进行更新：

```cpp
unsigned UpdateState_Literal(unsigned state)
{
  if (state < 4) return 0;
  else if (state < 10) return state - 3;
  else return state - 6;
}
unsigned UpdateState_Match   (unsigned state) { return state < 7 ? 7 : 10; }
unsigned UpdateState_Rep     (unsigned state) { return state < 7 ? 8 : 11; }
unsigned UpdateState_ShortRep(unsigned state) { return state < 7 ? 9 : 11; }
```

解码器计算 `state2` 变量值以从 `IsMatch` 和 `IsRep0Long` 数组中选择确切的变量：

```cpp
unsigned posState = OutWindow.TotalPos & ((1 << pb) - 1);
unsigned state2 = (state << kNumPosBitsMax) + posState;
```

解码器使用以下代码流方案来选择确切的字面值 (LITERAL) 或匹配 (MATCH) 类型：

1.  `IsMatch[state2]` 解码
    * **0 -> 字面值 (Literal)**
    * **1 -> 匹配 (Match)**
        1.  `IsRep[state]` 解码
            * **0 -> 简单匹配 (Simple Match)**
            * **1 -> 重复匹配 (Rep Match)**
                1.  `IsRepG0[state]` 解码
                    * **0 -> 距离是 rep0**
                        1.  `IsRep0Long[state2]` 解码
                            * **0 -> 短重复匹配 (Short Rep Match)**
                            * **1 -> 重复匹配 0 (Rep Match 0)**
                    * **1 ->**
                        1.  `IsRepG1[state]` 解码
                            * **0 -> 重复匹配 1 (Rep Match 1)**
                            * **1 ->**
                                1.  `IsRepG2[state]` 解码
                                    * **0 -> 重复匹配 2 (Rep Match 2)**
                                    * **1 -> 重复匹配 3 (Rep Match 3)**

### 字面值 (LITERAL) 符号

--------------

如果使用 `IsMatch[state2]` 解码得到的值为 "0"，则我们有 "字面值 (LITERAL)" 类型。

首先，LZMA 解码器必须检查它是否超过了指定的未压缩大小：

```cpp
if (unpackSizeDefined && unpackSize == 0) {
  return LZMA_RES_ERROR; // 返回：错误
}
```

然后它解码字面值并将其放入滑动窗口：

```cpp
DecodeLiteral(state, rep0);
```

然后解码器必须更新 `state` 值和 `unpackSize` 值；

```cpp
state = UpdateState_Literal(state);
unpackSize--;
```

然后解码器必须转到主循环的开始处以解码下一个匹配 (Match) 或字面值 (Literal)。

### 简单匹配 (Simple Match)

------------

如果使用 `IsMatch[state2]` 解码得到的值为 "1"，并且使用 `IsRep[state]` 解码得到的值为 "0"，我们有 "简单匹配 (Simple Match)" 类型。
距离历史表使用以下方案更新：

```cpp
rep3 = rep2;
rep2 = rep1;
rep1 = rep0;
```

使用 `LenDecoder` 解码从零开始的长度：

```cpp
len = LenDecoder.Decode(&RangeDec, posState);
```

使用 `UpdateState_Match` 函数更新状态：

```cpp
state = UpdateState_Match(state);
```

并使用 `DecodeDistance` 解码新的 `rep0` 值：

```cpp
rep0 = DecodeDistance(len);
```

这个 `rep0` 将用作当前匹配的从零开始的距离。

如果 `rep0` 的值等于 `0xFFFFFFFF`，则表示我们遇到了 "流结束 (End of stream)" 标记，因此我们可以停止解码并在范围解码器中检查结束条件：

```cpp
if (rep0 == 0xFFFFFFFF) {
  return RangeDec.IsFinishedOK() ?
      LZMA_RES_FINISHED_WITH_MARKER : // 返回：带标记完成
      LZMA_RES_ERROR;                 // 返回：错误
}
```

如果未压缩大小已定义，LZMA 解码器必须检查它是否超过了该指定的未压缩大小：

```cpp
if (unpackSizeDefined && unpackSize == 0)
  return LZMA_RES_ERROR; // 返回：错误
```

此外，解码器必须检查 `rep0` 值不大于字典大小，并且不大于已解码的字节数：

```cpp
if (rep0 >= dictSize || !OutWindow.CheckDistance(rep0))
  return LZMA_RES_ERROR; // 返回：错误
```

然后解码器必须按照 "匹配符号复制" 部分所述复制匹配字节。

### 重复匹配 (Rep Match)

---------

如果 LZMA 解码器使用 `IsRep[state]` 变量解码得到的值为 "1"，我们有 "重复匹配 (Rep Match)" 类型。

首先，LZMA 解码器必须检查它是否超过了指定的未压缩大小：

```cpp
if (unpackSizeDefined && unpackSize == 0)
  return LZMA_RES_ERROR; // 返回：错误
```

此外，如果 LZ 窗口为空，解码器必须返回错误：

```cpp
if (OutWindow.IsEmpty())
  return LZMA_RES_ERROR; // 返回：错误
```

如果匹配类型是 "重复匹配 (Rep Match)"，解码器使用距离历史表中的 4 个变量之一来获取当前匹配的距离值，并且有 4 种相应的解码流程。

根据匹配类型，解码器使用以下方案更新距离历史：

* **"重复匹配 0 (Rep Match 0)" 或 "短重复匹配 (Short Rep Match)"**:

    * `;` (LZMA 不更新距离历史)

* **"重复匹配 1 (Rep Match 1)"**:

    ```cpp
    UInt32 dist = rep1;
    rep1 = rep0;
    rep0 = dist;
    ```

* **"重复匹配 2 (Rep Match 2)"**:

    ```cpp
    UInt32 dist = rep2;
    rep2 = rep1;
    rep1 = rep0;
    rep0 = dist;
    ```

* **"重复匹配 3 (Rep Match 3)"**:

    ```cpp
    UInt32 dist = rep3;
    rep3 = rep2;
    rep2 = rep1;
    rep1 = rep0;
    rep0 = dist;
    ```

然后解码器使用 `IsRepG0`、`IsRep0Long`、`IsRepG1`、`IsRepG2` 解码 "重复匹配 (Rep Match)" 的确切子类型。

如果子类型是 "短重复匹配 (Short Rep Match)"，解码器更新状态，将窗口中的一个字节放到窗口中的当前位置，然后转到下一个 匹配 (MATCH) / 字面值 (LITERAL) 符号（主循环的开始）：

```cpp
state = UpdateState_ShortRep(state);
OutWindow.PutByte(OutWindow.GetByte(rep0 + 1));
unpackSize--;
continue;
```

在其他情况下（重复匹配 0/1/2/3），它使用 `RepLenDecoder` 解码器解码匹配的从零开始的长度：

```cpp
len = RepLenDecoder.Decode(&RangeDec, posState);
```

然后它更新状态：

```cpp
state = UpdateState_Rep(state);
```

然后解码器必须按照 "匹配符号复制" 部分所述复制匹配字节。

### 匹配符号复制

-------------------------

如果我们有一个匹配（简单匹配或重复匹配 0/1/2/3），解码器必须使用计算出的匹配距离和匹配长度复制字节序列。

如果未压缩大小已定义，LZMA 解码器必须检查它是否超过了该指定的未压缩大小：

```cpp
len += kMatchMinLen;
bool isError = false;
if (unpackSizeDefined && unpackSize < len) {
  len = (unsigned)unpackSize;
  isError = true;
}

OutWindow.CopyMatch(rep0 + 1, len);
unpackSize -= len;
if (isError) {
  return LZMA_RES_ERROR; // 返回：错误
}
```

然后解码器必须转到主循环的开始处以解码下一个 匹配 (MATCH) 或 字面值 (LITERAL)。

## 注意

-----

本规范未描述支持部分解码的解码器实现变体。

这种部分解码情况可能需要在 "流结束" 条件检查代码中进行一些更改。此类代码也可以使用解码器返回的附加状态码。

本规范使用带有模板的 C++ 代码来简化描述。

LZMA 解码器的优化版本不需要模板。这种优化版本可以只使用两个 `CProb` 变量数组：

1) 为字面值解码器分配的 `CProb` 变量的动态数组。
2) 一个包含所有其他 `CProb` 变量的公共数组。

## 参考文献：

1.  G. N. N. Martin, Range encoding: an algorithm for removing redundancy from a digitized message, Video & Data Recording Conference, Southampton, UK, July 24-27, 1979.
