
# Google 开源 NLP 模型 BERT

## 1. 简介

### 1.1 详解 `BERT`

这种新的语言表征模型 `BERT`，意思是来自 `Transformer` 的双向编码器表征（`Bidirectional Encoder Representations from Transformers`）。

`GitHub` 项目地址：[https://github.com/google-research/bert](https://github.com/google-research/bert)

`BERT` 论文原文：[https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805)

### 1.2 横扫 11 大 `NLP` 测试

机器阅读理解，是一场竞争激烈的比拼。

竞技场是 `SQuAD`。

`SQuAD` 是行业公认的机器阅读理解顶级水平测试，可以理解为机器阅读理解领域的 `ImageNet`。它们同样出自斯坦福，同样是一个数据集，搭配一个竞争激烈的竞赛。

这个竞赛基于 `SQuAD` 问答数据集，考察两个指标：`EM` 和 `F1`。

`EM` 是指 `精确匹配`，也就是模型给出的答案与标准答案一模一样；`F1`，是根据模型给出的答案和标准答案之间的 `重合度` 计算出来的，也就是结合了 `召回率` 和 `精确率`。

这次 `Google AI` 基于 `BERT` 的混合模型，在 `SQuAD 1.1` 数据集上，获得 `EM`：87.433、`F1`：93.160 分的历史最佳成绩。

## 2. 参考文献

1. `[NLP 自然语言处理]谷歌 BERT 模型深度解析`

    [https://blog.csdn.net/qq_39521554/article/details/83062188](https://blog.csdn.net/qq_39521554/article/details/83062188)

2. `全面超越人类！Google 称霸 SQuAD，BERT 横扫11大 NLP 测试`

    [https://zhuanlan.zhihu.com/p/46648916](https://zhuanlan.zhihu.com/p/46648916)

3. `谷歌最强 NLP 模型 BERT 解读`

    [https://www.leiphone.com/news/201810/KBOOk5ovADXhS0s0.html](https://www.leiphone.com/news/201810/KBOOk5ovADXhS0s0.html)

4. `机器这次击败人之后，争论一直没平息 | SQuAD风云`

    [https://mp.weixin.qq.com/s?__biz=MzIzNjc1NzUzMw==&mid=2247493419&idx=1&sn=73425fec04482f14f6b9b7316e425e63&chksm=e8d05059dfa7d94fc1457a36d4f62cb1b8a057ce18388fbad448aa6b53f4dbb1299cfd697724&scene=21#wechat_redirect](https://mp.weixin.qq.com/s?__biz=MzIzNjc1NzUzMw==&mid=2247493419&idx=1&sn=73425fec04482f14f6b9b7316e425e63&chksm=e8d05059dfa7d94fc1457a36d4f62cb1b8a057ce18388fbad448aa6b53f4dbb1299cfd697724&scene=21#wechat_redirect)

