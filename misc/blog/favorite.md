
# 个人博客收藏

## 1. 技术博客

1.1 [Eric Fu 的博客：Coding Husky](https://ericfu.me/)

    总体评价：博客样式还不错，值得借鉴，部分文章也不错。

文章精选：

* [YugabyteDB 介绍](https://ericfu.me/yugabyte-db-introduction/#more)
* [YugabyteDB 介绍](https://zhuanlan.zhihu.com/p/102589603) on `知乎`

* [JIT 代码生成技术（一）表达式编译](https://ericfu.me/code-gen-of-expression/)

    `Spark 2.0` 的表达式编译。

* [处理海量数据：列式存储综述（存储篇）](https://ericfu.me/columnar-storage-overview-storage/)

    事务型数据库（OLTP）大多采用行式存储，分析型数据库（OLAP）则多数采用列式存储。列式存储最经典的应用是 `ClickHouse`，但该文并未提及，只提到了数据的编码和压缩，减少IO。`ClickHouse` 在此基础上，使用 `SIMD` 技术加速处理列数据，并且由于 `SIMD` 的特点，是按 `Chunk` 读取和处理列数据，效率很高。

* [G1 垃圾收集器](https://ericfu.me/g1-garbage-collector/)

    介绍 `JDK 9` 中的 `G1` (Garbage-First) 垃圾回收器。 `G1` 中最核心的两个概念：Region 和 Remember Set。

可参考的版权声明样式：

```
本文作者： Eric Fu
本文链接： https://ericfu.me/yugabyte-db-introduction/
版权声明： 本博客所有文章除特别声明外，均采用 (C) BY-NC-SA 许可协议。转载请注明出处！
```

