# Terark 文档收集

## 1. 相关资料

1. Terark 公司简介

    `Terark 奇简软件` 是提供高压缩存储和高性能检索技术的公司。成立于2015年11月，获得Launch Hill与道合资本的投资。 Terark已拥有自主发明的可检索压缩SeComp技术、索引技术、手机检索技术等六项国内、国际专利。陆续发布了应用于云、数据库、手机等领域的Terark存储引擎、数据库、多正则匹配引擎、移动端检索引擎产品。其中，TerarkDB产品性能已经超越Facebook、Google、Berkeley等同类产品，能够为大数据应用降低至少50%成本的同时提高10倍性能。Terark为移动端提供超高性能检索框架，提高移动端本地检索效率20倍以上。Terark移动端使用人工智能技术对图像进行深度的理解，可以搜素图像中的文字(OCR)和人物等，开启下一个搜索时代。 Terark团队成员主要来自于Yahoo、Google、Microsoft、Baidu等知名企业，技术专家占比85%，具有技术发展前瞻性与强大的技术研发能力。Terark曾为猎豹、新浪等早期客户提供服务，现为阿里云核心数据技术供应商，以及京东OCR技术供应商。 Terark作为中国本土技术创业公司，获得仅有千分之一通过率的美国硅谷著名孵化器Y Combinator的严格评估筛选，加入YC训练营。国内，获得首都青年创业大赛一等奖。 Terark奇简软件致力于让数据更小、访问更快、更智能。Terark会继续发展服务器端和移动端技术，基于核心的存储、检索、人工智能技术优势，提供云、数据库、手机智能检索等产品。目前已成功建立可持续的盈利模式，更将通过技术革新与应用，助力企业规模化增长，为其提供可持续的价值增值。

    ```bash
    2019年04月29日 获得 字节跳动 并购
    2016年12月22日 获得 合力投资EmpowerInvestment 天使投资
    ```

    网址：[https://newseed.pedaily.cn/data/project/84846](https://newseed.pedaily.cn/data/project/84846)

## 2. 相关文档

1. [全局压缩-革命性的数据库技术](http://nark.cc/p/?p=1720) From `nark.cc`

    作者: rockeet，发表日期: 2017年03月08日

2. [TerarkDB - 我们发布了一款划时代的存储引擎](https://zhuanlan.zhihu.com/p/21493877) From `zhihu.com`

    作者：郭宽，发表日期：大约 2017-05-24

    目前世界上绝大多数开源数据库的索引都是由 `B+树` 或 `LSM` 实现的，但是 `Terark` 实现了一种完全不同的索引结构（`Succinct Trie`），可以把读性能提升一到两个数量级。

    `TerarkDB` 作为一个存储引擎，可以嵌入MongoDB、MySQL、SSDB等现有的存储系统中，也可以直接作为独立的存储系统进行使用。

3. [使用TerarkDB提升MySQL性能和压缩率](https://myslide.cn/slides/5559) From `myslide.cn`

    主讲人：Terark 联合创始人 郭宽宽

    作者：闫盼晴，发表于：2017/11/26

    已收录在本文档库中，路径为：\tech\db\terark\使用TerarkDB提升MySQL性能和压缩率.md

4. [TerarkDB数据库的性能报告与技术解析](https://www.2cto.com/database/201606/513841.html) From `www.2cto.com`

    作者：whinah (雷鹏)，2016-06-01

    TerarkDB 是一个拥有极高性能和数据压缩率的存储引擎。使用方法类似Facebook的RocksDB，不过比 RocksDB 具有更多功能，下面是 TerarkDB 的功能特性：

    高压缩率，通常是 snappy 的2～5倍 实时免解压直接检索数据 Query 延迟很低并且很稳定 同一 Table 可包含多个索引，支持联合索引，支持范围搜索 原生支持正则表达式检索 支持嵌入进程，或者 Server-Client 模式 数据持久化 支持 Schema，包含丰富的数据类型 列存储以及行存储，支持 Column Group。
