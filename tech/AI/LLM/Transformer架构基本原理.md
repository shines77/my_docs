# Transformer 架构基本原理

## 1. 概述

`Transformer` 模型 2017 年出自于 Google Brain 研究小组 Ashish Vaswani 等人发布的论文《Attention is all you need》中，是一种在自然语言处理（NLP）及其他序列到序列（Seq2Seq）任务中广泛使用的深度学习模型框架。

以下是小组各成员的贡献，名单顺序随机。Jakob 建议以 self-attention 取代 RNN，并开始努力评估这一想法。Ashish 与 Illia 一起设计并实现了第一个 Transformer 模型，并在这项工作中的各个方面起着至关重要的作用。Noam 提出了 scaled dot-product attention, multi-head attention 和参数无关的位置表示，并成为涉及几乎每个细节的另一个人。Niki 在我们原始的代码库和 tensor2tensor 中设计、实现、调优和评估了无数模型变体。Llion 还尝试了新的模型变体，负责我们的初始代码库以及高效的推理和可视化。Lukasz 和 Aidan 花了无数漫长的时间来设计和实现 tensor2tensor 的各个部分，以取代我们之前的代码库，从而大大改善了结果并极大地加速了我们的研究。

`Transformer` 架构最初旨在用于训练语言翻译模型，然而，2018 年 OpenAI 团队发现，`Transformer` 架构是字符预测的关键解决方案。一旦对整个互联网数据进行了训练，该模型就有可能理解任何文本的上下文，并连贯地完成任何句子，就像人类一样。

## 2. 结构

`Transformer` 模型由两部分组成：编码器 (encoder) 和解码器 (decoder)。一般来说，仅编码器 (encoder-only) 架构擅长从文本中提取信息，用于分类和回归等任务，而仅解码器 (decoder-only) 模型专门用于生成文本。例如，专注于文本生成的 GPT 属于仅解码器 (decoder-only) 模型的范畴。

## x. 参考文章

- [LLM: From Zero to Hero: Transformer Architecture](https://waylandzhang.github.io/en/transformer-architecture.html)

- [《Attention is all you need》论文及译文](https://xueqiu.com/3993902801/284722170)
