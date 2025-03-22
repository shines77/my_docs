# Terark 文档收集

## 1. 相关资料

1. Terark 公司简介

    `Terark 奇简软件` 是提供高压缩存储和高性能检索技术的公司。

    成立于 2015 年 11 月，获得 Launch Hill 与道合资本的投资。

    Terark 已拥有自主发明的可检索压缩 SeComp 技术、索引技术、手机检索技术等六项国内、国际专利。陆续发布了应用于云、数据库、手机等领域的 Terark 存储引擎、数据库、多正则匹配引擎、移动端检索引擎产品。其中，TerarkDB 产品性能已经超越 Facebook、Google、Berkeley 等同类产品，能够为大数据应用降低至少 50% 成本的同时提高 10 倍性能。Terark 为移动端提供超高性能检索框架，提高移动端本地检索效率 20 倍以上。Terark移动端使用人工智能技术对图像进行深度的理解，可以搜素图像中的文字(OCR)和人物等，开启下一个搜索时代。

    Terark 团队成员主要来自于 Yahoo、Google、Microsoft、Baidu 等知名企业，技术专家占比85%，具有技术发展前瞻性与强大的技术研发能力。Terark 曾为猎豹、新浪等早期客户提供服务，现为阿里云核心数据技术供应商，以及京东 OCR 技术供应商。

    Terark 作为中国本土技术创业公司，获得仅有千分之一通过率的美国硅谷著名孵化器 Y Combinator 的严格评估筛选，加入 YC 训练营。国内，获得首都青年创业大赛一等奖。

    Terark 奇简软件致力于让数据更小、访问更快、更智能。Terark 会继续发展服务器端和移动端技术，基于核心的存储、检索、人工智能技术优势，提供云、数据库、手机智能检索等产品。目前已成功建立可持续的盈利模式，更将通过技术革新与应用，助力企业规模化增长，为其提供可持续的价值增值。

    ```bash
    2019年04月29日 获得 字节跳动 并购
    2016年12月22日 获得 合力投资 EmpowerInvestment 天使投资
    ```

    网址：[https://newseed.pedaily.cn/data/project/84846](https://newseed.pedaily.cn/data/project/84846)

## 2. 相关文档

1. [全局压缩--革命性的数据库技术](http://nark.cc/p/?p=1720) From `nark.cc`

    作者：rockeet，发表日期: 2017年03月08日

2. [TerarkDB - 我们发布了一款划时代的存储引擎](https://zhuanlan.zhihu.com/p/21493877) From `zhihu.com`

    作者：郭宽，发表日期：大约 2017-05-24

    目前世界上绝大多数开源数据库的索引都是由 `B+树` 或 `LSM` 实现的，但是 `Terark` 实现了一种完全不同的索引结构（`Succinct Trie`），可以把读性能提升一到两个数量级。

    `TerarkDB` 作为一个存储引擎，可以嵌入MongoDB、MySQL、SSDB等现有的存储系统中，也可以直接作为独立的存储系统进行使用。

3. [使用 TerarkDB 提升 MySQL 性能和压缩率](https://myslide.cn/slides/5559) From `myslide.cn`

    主讲人：Terark 联合创始人 郭宽宽

    作者：闫盼晴，发表于：2017/11/26

    已收录在本文档库中，路径为：\tech\db\terark\使用TerarkDB提升MySQL性能和压缩率.md

4. [TerarkDB 数据库的性能报告与技术解析](https://blog.csdn.net/whinah/article/details/51545839) From `blog.csdn.net`

    作者：whinah (雷鹏)，2016-05-31

    TerarkDB 是一个拥有极高性能和数据压缩率的存储引擎。使用方法类似 Facebook 的 RocksDB，不过比 RocksDB 具有更多功能，下面是 TerarkDB 的功能特性：

    高压缩率，通常是 snappy 的 2～5 倍 实时免解压直接检索数据 Query 延迟很低并且很稳定，同一 Table 可包含多个索引，支持联合索引，支持范围搜索 原生支持正则表达式检索 支持嵌入进程，或者 Server-Client 模式 数据持久化 支持 Schema，包含丰富的数据类型 列存储以及行存储，支持 Column Group。

5. [Succinct 技术](https://zhuanlan.zhihu.com/p/362110145) From `知乎`

    作者：[诚毅](https://www.zhihu.com/people/xu-liang-57-86)，2021-04-14

    - 传统数据库中的块压缩技术

        zip、gzip、bzip2、Snappy、LZ4、Zstd 等。按块/页（block/page）进行压缩（块尺寸通常是 4KB ~ 32KB，以压缩率著称的 TokuDB 块尺寸是 2MB ~ 4MB）。

    - RocksDB 的块压缩

        RocksDB 中的 BlockBasedTable 就是一个块压缩的 SSTable，使用块压缩，索引只定位到块，块的尺寸在 dboption 里设定，一个块中包含多条（key，value）数据。

    - 传统非主流压缩：FM-Index

        FM-Index 的全名是 Full Text Matching Index，属于 Succinct Data Structure 家族。Berkeley 大学的 Succinct 项目（Java）也使用了 FM-Index。

    - Terark 的可检索压缩（Searchable Compression）

        Terark 公司提出了“可检索压缩（Searchable Compression）”的概念，其核心也是直接在压缩的数据上执行搜索（search）和访问（extract），但数据模型本身就是 Key-Value 模型。

6. [数据库压缩技术探索](https://www.51cto.com/article/542294.html)

    作者：雷鹏，2017-06-12

    内容基本跟上文一样，但更详细，这个才是原文。
