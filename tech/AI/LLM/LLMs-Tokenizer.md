# LLMs Tokenizer

## 1. Byte-Pair Encoding

字节对编码 (Byte-Pair Encoding, BPE) 最初是作为一种压缩文本的算法开发的，后来被 OpenAI 用于预训练 GPT 模型时的分词 (tokenization) 过程。现在，许多 Transformer 模型都在使用它，包括 GPT、GPT-2、RoBERTa、BART 和 DeBERTa 。

BPE 算法的核心思想是：将文本中的 频繁出现的字节对 替换为一个新的 字节对。通过不断迭代这个过程，最终将文本编码成一系列的字节或字节对。

BPE 算法的具体步骤如下：

1. 统计文本中所有字节对出现的频率。
2. 找到出现频率最高的字节对。
3. 将该字节对替换为一个新的字节。
4. 重复步骤 1-3，直到满足终止条件。

终止条件可以是：

- 文本中所有字节对的出现频率都低于某个阈值。
- 文本的编码长度达到某个要求。

BPE 算法具有以下优势：

- 分词效果好，能够准确地识别词语和符号的边界。
- 编码效率高，能够节省空间。
- 能够将词语或符号之间的关系编码到编码中，有利于模型学习。

我个人认为 BPE 抹杀了 Word 作为自然语言的较小的一个单位作为语义上的表达，Word 可以拆分只能是因为词缀，而不能强行拆分。BPE 并不是一个完美的编码方式。

## 2. Byte-Lvel Encoding



## 3. WordPiece

WordPiece，从名字好理解，它是一种子词粒度的 tokenize 算法 subword tokenization algorithm，很多著名的 Transformers 模型，比如 BERT / DistilBERT / Electra 都使用了它。

它的原理非常接近 BPE，不同之处在于它做合并时，并不是直接找最高频的组合，而是找能够最大化训练数据似然的 merge 。即它每次合并的两个字符串 A 和 B，应该具有最大的 $\frac{P(AB)}{P(A) \cdot P(B)}$ 值。合并 AB 之后，所有原来切成 A+B 两个 tokens 的就只保留 AB 一个 token，整个训练集上最大似然变化量与 $\frac{P(AB)}{P(A) \cdot P(B)}$ 成正比。

## 4. Unigram

与 BPE 或者 WordPiece 不同，Unigram 的算法思想是从一个巨大的词汇表出发，再逐渐删除 trim down 其中的词汇，直到 size 满足预定义。

初始的词汇表可以采用所有预分词器分出来的词，再加上所有高频的子串。每次从词汇表中删除词汇的原则是使预定义的损失最小。

Unigram 算法每次会从词汇表中挑出使得 loss 增长最小的 10% ~ 20% 的词汇来删除。一般 Unigram 算法会与 SentencePiece 算法连用。

## 5. SentencePiece

SentencePiece 则是基于 `无监督学习` 的，用的是一种称为 Masked Language Modeling (MLM) 的算法。MLM 的基本思想是：将文本中的部分词语或符号进行遮蔽，然后让模型预测被遮蔽的词语或符号，通过这种方式，模型可以学习文本的语义和结构。

顾名思义，它是把一个句子看作一个整体，再拆成片段，而没有保留天然的词语的概念。一般地，它把空格 space 也当作一种特殊字符来处理，再用 BPE 或者 Unigram 算法来构造词汇表。

比如，XLNetTokenizer 就采用了 _ 来代替空格，解码的时候会再用空格替换回来。

MLM 的具体实现如下：

- 随机选择文本中的部分词语或符号进行遮蔽。
- 将被遮蔽的词语或符号替换为一个特殊符号，例如 [MASK]。
- 将处理后的文本输入模型，让模型预测被遮蔽的词语或符号。

目前，Tokenizers 库中，所有使用了 SentencePiece 的都是与 Unigram 算法联合使用的，比如 ALBERT、XLNet、Marian 和 T5 。

**使用建议**：

- 如果需要对中文文本进行分词，并且对分词效果要求较高，可以选择 SentencePiece、Jieba 或 Hmmseg。
- 如果需要对多种语言文本进行分词，可以选择 Stanford CoreNLP。
- 如果需要对文本进行分词和编码，并且对速度要求较高，可以选择 Jieba。
- 如果需要对文本进行分词和编码，并且对分词效果要求较高，可以选择 SentencePiece 或 Hmmseg。

## 6. 参考文章

- LLMs Tokenizer

  - [大模型分词：sentencepiece vs tiktoken](https://zhuanlan.zhihu.com/p/691609961)

  - [BPE、WordPiece 和 SentencePiece](https://www.jianshu.com/p/d4de091d1367)

  - [Byte-Pair Encoding 分词算法速读](https://zhuanlan.zhihu.com/p/701869443)

  - [Byte-Pair Encoding(BPE) 分词算法详解](https://zhuanlan.zhihu.com/p/716655053)

  - [Byte-Pair Encoding 算法超详细讲解](https://www.jianshu.com/p/865b741f7b96)

  - [一文读懂：词向量 Word2Vec](https://zhuanlan.zhihu.com/p/371147732)
