# High performance hash table

## 1. Vedios

1. `[C++Now 2018: You Can Do Better than std::unordered_map: New Improvements to Hash Table Performance]`

    [https://www.youtube.com/watch?v=M2fKMP47slQ](https://www.youtube.com/watch?v=M2fKMP47slQ)

## 2. Article

1. `[Swiss Tables Design Notes]`

    [https://abseil.io/about/design/swisstables](https://abseil.io/about/design/swisstables)

## 3. Pref test

1. `HashTable性能测试(CK/phmap/ska)`

    [https://dirtysalt.github.io/html/hashtable-perf-comparison.html](https://dirtysalt.github.io/html/hashtable-perf-comparison.html)

    仅测试了 HashSet，没有测试 HashMap。

    单机单线程，插入 65536000 行，然后基数情况如下，都是插入随机数

    * [A] 960，低基数
    * [B] 96000，中基数
    * [C] 9600000，高基数
    * [D] 960000000，超高基数

    测试：

    * ska::flat_hash_map
    * CK: ClickHouse 分离出来的哈希表

    其中：

    * run_insert_random 表示 phmap
    * run_insert_random_ska 表示 ska hashmap
    * run_insert_precompute 表示 phmap 预先计算 hashvalue + prefetch
    * run_insert_random_ck 表示 CK，ClickHouse 分离出来的哈希表
