# High performance hash table

## 1. Vedios

1. `Matt Kulukundis: [CppCon 2017: "Designing a Fast, Efficient, Cache-friendly Hash Table, Step by Step"]`

    [https://www.youtube.com/watch?v=ncHmEUmJZf4](https://www.youtube.com/watch?v=ncHmEUmJZf4)

2. `Malte Skarupke: [C++Now 2018: You Can Do Better than std::unordered_map: New Improvements to Hash Table Performance]`

    [https://www.youtube.com/watch?v=M2fKMP47slQ](https://www.youtube.com/watch?v=M2fKMP47slQ)

## 2. Article

1. `[Swiss Tables Design Notes]`

    [https://abseil.io/about/design/swisstables](https://abseil.io/about/design/swisstables)

## 3. Benchmark

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

2. `martinus: map_benchmark`

    [https://github.com/martinus/map_benchmark](https://github.com/martinus/map_benchmark)

    包含下列 hashmap:

    * `stl`: std::unordered_map
    * `Boost`: boost::unordered_map, boost::multi_index::hashed_unique
    * `Google`: absl::flat_hash_map, absl::node_hash_map
    * `EASTL`: eastl::hash_map
    * `Facebook Folly`: folly::F14ValueMap, folly::F14NodeMap
    * `greg7mdp`’s parallel hashmap: phmap::flat_hash_map, phmap::node_hash_map
    * `ktprime`’s HashMap: emilib1::HashMap
    * `martinus`’s robin-hood-hashing: robin_hood::unordered_flat_map, robin_hood::unordered_node_map
    * `Malte Skarupke`’s HashMap: ska::flat_hash_map, ska::bytell_hash_map
    * `tessil`’s HashMap: tsl::hopscotch_map, tsl::robin_map, tsl::sparse_map

3. `martinus: Hashmaps Benchmarks - Overview`

    [https://martin.ankerl.com/2019/04/01/hashmap-benchmarks-01-overview/](https://martin.ankerl.com/2019/04/01/hashmap-benchmarks-01-overview/)

    `martinus: map_benchmark` 的一个 benchmark 结果（图表），大约于 2019 年测试。

    包含的 `hashmap` 同上。

4. `k-nucleotide Game`

    description: [https://benchmarksgame-team.pages.debian.net/benchmarksgame/description/knucleotide.html](https://benchmarksgame-team.pages.debian.net/benchmarksgame/description/knucleotide.html#knucleotide)

    C++ source: [https://benchmarksgame-team.pages.debian.net/benchmarksgame/program/knucleotide-gpp-2.html](https://benchmarksgame-team.pages.debian.net/benchmarksgame/program/knucleotide-gpp-2.html)

    `k-nucleotide` input data:

    fasta description: [https://benchmarksgame-team.pages.debian.net/benchmarksgame/description/fasta.html](https://benchmarksgame-team.pages.debian.net/benchmarksgame/description/fasta.html#fasta)

    `k-nucleotide` game in `Malte Skarupke` C++Now 2018 vedio
