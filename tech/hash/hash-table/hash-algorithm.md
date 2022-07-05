# 哈希表算法相关资料

## 1. 哈希函数

1. 斐波那契散列法:

    ```cpp
    hash = (value * 2654435769) >> 28;
    ```

2. `eastl` 中的哈希函数:

    ```cpp
    hash = 2166136261
    hash = (hash * 16777619) ^ value;
    ```

------------------------------------------

分布式技术探索——如何判断哈希的好坏

[https://zhuanlan.zhihu.com/p/53710382](https://zhuanlan.zhihu.com/p/53710382)

先来看看一下 `DJB` :

```cpp
unsigned long hash(unsigned char *str)
{
    unsigned long hash = 5381;
    int c;

    while (c = *str++) {
        hash = ((hash << 5) + hash) + c; /* hash * 33 + c */
    }

    return hash;
}
```

再来看看 `FNV` 哈希算法:

```cpp
template <class T>
size_t FNVHash(const T* str)
{
    if (!*str) {
        return 0;
    }
    size_t hash = 2166136261;   // Seed
    while (size_t ch = (size_t)*str++) {
        hash *= 16777619;
        hash ^= ch;
    }
    return hash;
}
```

我们继续看一个哈希算法, `BKDRHash` :

```cpp
unsigned int bkdr_hash(const char * key)
{
    char * str = const_cast<char *>(key);
    unsigned int seed = 31; // 31 131 1313 13131 131313 etc.. 37
    unsigned int hash = 0;
    while (*str) {
        hash = hash * seed + (*str++);
    }
    return hash;
}
```

.

深入解析面向数据的哈希表性能

https://zhuanlan.zhihu.com/p/26417610

我测试了四个不同的 `quick-and-dirty` 哈希表实现，另外还包括 `std::unordered_map` 。这五个哈希表都使用了同一个哈希函数 —— `Bob Jenkins` 的 `SpookyHash`（64 位哈希值）。（由于哈希函数在这里不是重点，所以我没有测试不同的哈希函数；我同样也没有检测我的分析中的总内存消耗。）实现会通过简短的代码在测试结果表中标注出来。

UM： `std::unordered_map` 。在 `VS2012` 和 `libstdc++-v3` （libstdc++-v3: gcc 和 clang 都会用到这东西）中，UM 是以链表的形式实现，所有的元素都在链表中，bucket 数组中存储了链表的迭代器。VS2012 中，则是一个双链表，每一个 bucket 存储了起始迭代器和结束迭代器；`libstdc++` 中，是一个单链表，每一个 bucket 只存储了一个起始迭代器。这两种情况里，链表节点是独立申请和释放的。最大负载因子是 1 。

```text
插入       insert: 将一个随机的 key 序列插入到表中（key 在序列中是唯一的）
预留插入   presized insert: 和 insert 相似，但是在插入之间我们先为所有的 key 保留足够的内存空间，以防止在 insert 过程中 rehash 或者重申请。

查询       find
失败查询   failed find

删除       erase
失败删除   failed erase

扩容       grow

重新hash   rehash

析构       destruct
```

```cpp
/*
* My best guess at if you are big-endian or little-endian.
* This may need adjustment.
*/
#if (defined(__BYTE_ORDER) && defined(__LITTLE_ENDIAN) && \
     __BYTE_ORDER == __LITTLE_ENDIAN) || \
    (defined(i386) || defined(__i386__) || defined(__i486__) || \
     defined(__i586__) || defined(__i686__) || defined(vax) || defined(MIPSEL))
# define HASH_LITTLE_ENDIAN 1
# define HASH_BIG_ENDIAN 0
#elif (defined(__BYTE_ORDER) && defined(__BIG_ENDIAN) && \
       __BYTE_ORDER == __BIG_ENDIAN) || \
      (defined(sparc) || defined(POWERPC) || defined(mc68000) || defined(sel))
# define HASH_LITTLE_ENDIAN 0
# define HASH_BIG_ENDIAN 1
#else
# define HASH_LITTLE_ENDIAN 0
# define HASH_BIG_ENDIAN 0
```

Hash 函数概览

本文地址：[https://www.oschina.net/translate/state-of-hash-functions](https://www.oschina.net/translate/state-of-hash-functions)

原文地址：[http://blog.reverberate.org/2012/01/state-of-hash-functions-2012.html](http://blog.reverberate.org/2012/01/state-of-hash-functions-2012.html)

现代的 Hash 算法

`Bob Jenkins' Functions`

`Bob Jenkins` 已经在散列函数领域工作了将近15年。在1997年他在《 Dr. Dobbs Journal》杂志上发表了一片关于散列函数的文章《A hash function for hash Table lookup》，这篇文章自从发表以后现在网上有更多的扩展内容。这篇文章中，Bob广泛收录了很多已有的散列函数，这其中也包括了他自己所谓的“lookup2”。随后在2006年，Bob发布了lookup3，由于它即快速（Bob自称，0.5 bytes/cycle）又无严重缺陷，在这篇文章中我把它认为是第一个“现代”散列函数。

更多有关 `Bob's 散列函数` 的内容请参阅维基百科：`Jenkins hash function`.

第二代: `MurmurHash`

Austin Appleby在2008年发布了一个新的散列函数——MurmurHash。其最新版本大约是lookup3速度的2倍（大约为1 byte/cycle），它有32位和64位两个版本。32位版本只使用32位数学函数并给出一个32位的哈希值，而64位版本使用了64位的数学函数，并给出64位哈希值。根据Austin的分析，MurmurHash具有优异的性能，虽然Bob Jenkins 在《Dr. Dobbs article》杂志上声称“我预测[MurmurHash ]比起lookup3要弱，但是我不知道具体值，因为我还没测试过它”。MurmurHash能够迅速走红得益于其出色的速度和统计特性。

第三代: `CityHash` 和 `SpookyHash`

2011年，发布了两个散列函数，相对于MurmurHash ，它们都进行了改善，这主要应归功于更高的指令级并行机制。Google发布了CityHash（由Geoff Pike 和Jyrki Alakuijala编写），Bob Jenkins发布了他自己的一个新散列函数SpookyHash（这样命名是因为它是在万圣节发布的）。它们都拥有2倍于MurmurHash的速度，但他们都只使用了64位数学函数而没有32位版本，并且CityHash的速度取决于CRC32 指令，目前为SSE 4.2（Intel Nehalem及以后版本）。SpookyHash给出128位输出，而CityHash有64位，128位以及256位的几个变种。

`FarmHash`

[https://github.com/google/farmhash](https://github.com/google/farmhash)

2014年Google发布了FarmHash，一个新的用于字符串的哈希函数系列。FarmHash从CityHash继承了许多技巧和技术，是它的后继。FarmHash有多个目标，声称从多个方面改进了CityHash。与CityHash相比，FarmHash的另一项改进是在多个特定于平台的实现之上提供了一个接口。这样，当开发人员只是想要一个用于哈希表的、快速健壮的哈希函数，而不需要在每个平台上都一样时，FarmHash也能满足要求。目前，FarmHash只包含在32、64和128位平台上用于字节数组的哈希函数。未来开发计划包含了对整数、元组和其它数据的支持。

The hash methods are platform dependent. Different CPU architectures, for example 32-bit vs 64-bit, Intel vs ARM, SSE4.2 vs AVX might produce different results for a given input.

[https://www.biaodianfu.com/hash.html](https://www.biaodianfu.com/hash.html)

`SuperFastHash`

[http://www.azillionmonkeys.com/qed/hash.html](http://www.azillionmonkeys.com/qed/hash.html)

`XXHash`

[https://github.com/Cyan4973/xxHash](https://github.com/Cyan4973/xxHash)

[https://cyan4973.github.io/xxHash/](https://cyan4973.github.io/xxHash/)

.

为什么 Java String 哈希乘数为 31 ？

[https://juejin.im/post/6844903683361079309](https://juejin.im/post/6844903683361079309)

------------------------------------------------------

个人笔记本，Windows 7操作系统，酷睿i5双核64位CPU。

测试数据：CentOS Linux release 7.5.1804的 `/usr/share/dict/words` 字典文件对应的所有单词。

由于CentOS上找不到该字典文件，通过 `yum -y install words` 进行了安装。

`/usr/share/dict/words` 共有 `479828` 个单词，该文件链接的原始文件为 `linux.words` 。

`Ubuntu` 上装 `words` 的方法：

```shell
apt-get install --reinstall wamerican
```

[https://askubuntu.com/questions/149125/how-to-use-dict-words](https://askubuntu.com/questions/149125/how-to-use-dict-words)

.

比常规CRC32C算法快12倍的优化

https://bbs.csdn.net/topics/370011606

CRC-32C (Castagnoli) 算法是 iSCSI 和 SCTP 数据校验的算法，和常用 CRC-32-IEEE 802.3 算法所不同的是多项式常数 CRC32C 是 0x1EDC6F41, CRC32 是 0x04C11DB7, 也就是说由此生成的CRC表不同外算法是一模一样.

```cpp
// polynomial = 0x1EDC6F41 ??

static const uint32_t CRC32c_Table = {
    0x00000000, 0xF26B8303, 0xE13B70F7, 0x1350F3F4,
    0xC79A971F, 0x35F1141C, 0x26A1E7E8, 0xD4CA64EB,
    0x8AD958CF, 0x78B2DBCC, 0x6BE22838, 0x9989AB3B,
    0x4D43CFD0, 0xBF284CD3, 0xAC78BF27, 0x5E133C24,
    0x105EC76F, 0xE235446C, 0xF165B798, 0x030E349B,
    0xD7C45070, 0x25AFD373, 0x36FF2087, 0xC494A384,
    0x9A879FA0, 0x68EC1CA3, 0x7BBCEF57, 0x89D76C54,
    0x5D1D08BF, 0xAF768BBC, 0xBC267848, 0x4E4DFB4B,
    0x20BD8EDE, 0xD2D60DDD, 0xC186FE29, 0x33ED7D2A,
    0xE72719C1, 0x154C9AC2, 0x061C6936, 0xF477EA35,
    0xAA64D611, 0x580F5512, 0x4B5FA6E6, 0xB93425E5,
    0x6DFE410E, 0x9F95C20D, 0x8CC531F9, 0x7EAEB2FA,
    0x30E349B1, 0xC288CAB2, 0xD1D83946, 0x23B3BA45,
    0xF779DEAE, 0x05125DAD, 0x1642AE59, 0xE4292D5A,
    0xBA3A117E, 0x4851927D, 0x5B016189, 0xA96AE28A,
    0x7DA08661, 0x8FCB0562, 0x9C9BF696, 0x6EF07595,
    0x417B1DBC, 0xB3109EBF, 0xA0406D4B, 0x522BEE48,
    0x86E18AA3, 0x748A09A0, 0x67DAFA54, 0x95B17957,
    0xCBA24573, 0x39C9C670, 0x2A993584, 0xD8F2B687,
    0x0C38D26C, 0xFE53516F, 0xED03A29B, 0x1F682198,
    0x5125DAD3, 0xA34E59D0, 0xB01EAA24, 0x42752927,
    0x96BF4DCC, 0x64D4CECF, 0x77843D3B, 0x85EFBE38,
    0xDBFC821C, 0x2997011F, 0x3AC7F2EB, 0xC8AC71E8,
    0x1C661503, 0xEE0D9600, 0xFD5D65F4, 0x0F36E6F7,
    0x61C69362, 0x93AD1061, 0x80FDE395, 0x72966096,
    0xA65C047D, 0x5437877E, 0x4767748A, 0xB50CF789,
    0xEB1FCBAD, 0x197448AE, 0x0A24BB5A, 0xF84F3859,
    0x2C855CB2, 0xDEEEDFB1, 0xCDBE2C45, 0x3FD5AF46,
    0x7198540D, 0x83F3D70E, 0x90A324FA, 0x62C8A7F9,
    0xB602C312, 0x44694011, 0x5739B3E5, 0xA55230E6,
    0xFB410CC2, 0x092A8FC1, 0x1A7A7C35, 0xE811FF36,
    0x3CDB9BDD, 0xCEB018DE, 0xDDE0EB2A, 0x2F8B6829,
    0x82F63B78, 0x709DB87B, 0x63CD4B8F, 0x91A6C88C,
    0x456CAC67, 0xB7072F64, 0xA457DC90, 0x563C5F93,
    0x082F63B7, 0xFA44E0B4, 0xE9141340, 0x1B7F9043,
    0xCFB5F4A8, 0x3DDE77AB, 0x2E8E845F, 0xDCE5075C,
    0x92A8FC17, 0x60C37F14, 0x73938CE0, 0x81F80FE3,
    0x55326B08, 0xA759E80B, 0xB4091BFF, 0x466298FC,
    0x1871A4D8, 0xEA1A27DB, 0xF94AD42F, 0x0B21572C,
    0xDFEB33C7, 0x2D80B0C4, 0x3ED04330, 0xCCBBC033,
    0xA24BB5A6, 0x502036A5, 0x4370C551, 0xB11B4652,
    0x65D122B9, 0x97BAA1BA, 0x84EA524E, 0x7681D14D,
    0x2892ED69, 0xDAF96E6A, 0xC9A99D9E, 0x3BC21E9D,
    0xEF087A76, 0x1D63F975, 0x0E330A81, 0xFC588982,
    0xB21572C9, 0x407EF1CA, 0x532E023E, 0xA145813D,
    0x758FE5D6, 0x87E466D5, 0x94B49521, 0x66DF1622,
    0x38CC2A06, 0xCAA7A905, 0xD9F75AF1, 0x2B9CD9F2,
    0xFF56BD19, 0x0D3D3E1A, 0x1E6DCDEE, 0xEC064EED,
    0xC38D26C4, 0x31E6A5C7, 0x22B65633, 0xD0DDD530,
    0x0417B1DB, 0xF67C32D8, 0xE52CC12C, 0x1747422F,
    0x49547E0B, 0xBB3FFD08, 0xA86F0EFC, 0x5A048DFF,
    0x8ECEE914, 0x7CA56A17, 0x6FF599E3, 0x9D9E1AE0,
    0xD3D3E1AB, 0x21B862A8, 0x32E8915C, 0xC083125F,
    0x144976B4, 0xE622F5B7, 0xF5720643, 0x07198540,
    0x590AB964, 0xAB613A67, 0xB831C993, 0x4A5A4A90,
    0x9E902E7B, 0x6CFBAD78, 0x7FAB5E8C, 0x8DC0DD8F,
    0xE330A81A, 0x115B2B19, 0x020BD8ED, 0xF0605BEE,
    0x24AA3F05, 0xD6C1BC06, 0xC5914FF2, 0x37FACCF1,
    0x69E9F0D5, 0x9B8273D6, 0x88D28022, 0x7AB90321,
    0xAE7367CA, 0x5C18E4C9, 0x4F48173D, 0xBD23943E,
    0xF36E6F75, 0x0105EC76, 0x12551F82, 0xE03E9C81,
    0x34F4F86A, 0xC69F7B69, 0xD5CF889D, 0x27A40B9E,
    0x79B737BA, 0x8BDCB4B9, 0x988C474D, 0x6AE7C44E,
    0xBE2DA0A5, 0x4C4623A6, 0x5F16D052, 0xAD7D5351
};

// polynomial = 0xEDB88320L

static const uint32_t crc32tab[] = {
    0x00000000L, 0x77073096L, 0xee0e612cL, 0x990951baL,
    0x076dc419L, 0x706af48fL, 0xe963a535L, 0x9e6495a3L,
    0x0edb8832L, 0x79dcb8a4L, 0xe0d5e91eL, 0x97d2d988L,
    0x09b64c2bL, 0x7eb17cbdL, 0xe7b82d07L, 0x90bf1d91L,
    0x1db71064L, 0x6ab020f2L, 0xf3b97148L, 0x84be41deL,
    0x1adad47dL, 0x6ddde4ebL, 0xf4d4b551L, 0x83d385c7L,
    0x136c9856L, 0x646ba8c0L, 0xfd62f97aL, 0x8a65c9ecL,
    0x14015c4fL, 0x63066cd9L, 0xfa0f3d63L, 0x8d080df5L,
    0x3b6e20c8L, 0x4c69105eL, 0xd56041e4L, 0xa2677172L,
    0x3c03e4d1L, 0x4b04d447L, 0xd20d85fdL, 0xa50ab56bL,
    0x35b5a8faL, 0x42b2986cL, 0xdbbbc9d6L, 0xacbcf940L,
    0x32d86ce3L, 0x45df5c75L, 0xdcd60dcfL, 0xabd13d59L,
    0x26d930acL, 0x51de003aL, 0xc8d75180L, 0xbfd06116L,
    0x21b4f4b5L, 0x56b3c423L, 0xcfba9599L, 0xb8bda50fL,
    0x2802b89eL, 0x5f058808L, 0xc60cd9b2L, 0xb10be924L,
    0x2f6f7c87L, 0x58684c11L, 0xc1611dabL, 0xb6662d3dL,
    0x76dc4190L, 0x01db7106L, 0x98d220bcL, 0xefd5102aL,
    0x71b18589L, 0x06b6b51fL, 0x9fbfe4a5L, 0xe8b8d433L,
    0x7807c9a2L, 0x0f00f934L, 0x9609a88eL, 0xe10e9818L,
    0x7f6a0dbbL, 0x086d3d2dL, 0x91646c97L, 0xe6635c01L,
    0x6b6b51f4L, 0x1c6c6162L, 0x856530d8L, 0xf262004eL,
    0x6c0695edL, 0x1b01a57bL, 0x8208f4c1L, 0xf50fc457L,
    0x65b0d9c6L, 0x12b7e950L, 0x8bbeb8eaL, 0xfcb9887cL,
    0x62dd1ddfL, 0x15da2d49L, 0x8cd37cf3L, 0xfbd44c65L,
    0x4db26158L, 0x3ab551ceL, 0xa3bc0074L, 0xd4bb30e2L,
    0x4adfa541L, 0x3dd895d7L, 0xa4d1c46dL, 0xd3d6f4fbL,
    0x4369e96aL, 0x346ed9fcL, 0xad678846L, 0xda60b8d0L,
    0x44042d73L, 0x33031de5L, 0xaa0a4c5fL, 0xdd0d7cc9L,
    0x5005713cL, 0x270241aaL, 0xbe0b1010L, 0xc90c2086L,
    0x5768b525L, 0x206f85b3L, 0xb966d409L, 0xce61e49fL,
    0x5edef90eL, 0x29d9c998L, 0xb0d09822L, 0xc7d7a8b4L,
    0x59b33d17L, 0x2eb40d81L, 0xb7bd5c3bL, 0xc0ba6cadL,
    0xedb88320L, 0x9abfb3b6L, 0x03b6e20cL, 0x74b1d29aL,
    0xead54739L, 0x9dd277afL, 0x04db2615L, 0x73dc1683L,
    0xe3630b12L, 0x94643b84L, 0x0d6d6a3eL, 0x7a6a5aa8L,
    0xe40ecf0bL, 0x9309ff9dL, 0x0a00ae27L, 0x7d079eb1L,
    0xf00f9344L, 0x8708a3d2L, 0x1e01f268L, 0x6906c2feL,
    0xf762575dL, 0x806567cbL, 0x196c3671L, 0x6e6b06e7L,
    0xfed41b76L, 0x89d32be0L, 0x10da7a5aL, 0x67dd4accL,
    0xf9b9df6fL, 0x8ebeeff9L, 0x17b7be43L, 0x60b08ed5L,
    0xd6d6a3e8L, 0xa1d1937eL, 0x38d8c2c4L, 0x4fdff252L,
    0xd1bb67f1L, 0xa6bc5767L, 0x3fb506ddL, 0x48b2364bL,
    0xd80d2bdaL, 0xaf0a1b4cL, 0x36034af6L, 0x41047a60L,
    0xdf60efc3L, 0xa867df55L, 0x316e8eefL, 0x4669be79L,
    0xcb61b38cL, 0xbc66831aL, 0x256fd2a0L, 0x5268e236L,
    0xcc0c7795L, 0xbb0b4703L, 0x220216b9L, 0x5505262fL,
    0xc5ba3bbeL, 0xb2bd0b28L, 0x2bb45a92L, 0x5cb36a04L,
    0xc2d7ffa7L, 0xb5d0cf31L, 0x2cd99e8bL, 0x5bdeae1dL,
    0x9b64c2b0L, 0xec63f226L, 0x756aa39cL, 0x026d930aL,
    0x9c0906a9L, 0xeb0e363fL, 0x72076785L, 0x05005713L,
    0x95bf4a82L, 0xe2b87a14L, 0x7bb12baeL, 0x0cb61b38L,
    0x92d28e9bL, 0xe5d5be0dL, 0x7cdcefb7L, 0x0bdbdf21L,
    0x86d3d2d4L, 0xf1d4e242L, 0x68ddb3f8L, 0x1fda836eL,
    0x81be16cdL, 0xf6b9265bL, 0x6fb077e1L, 0x18b74777L,
    0x88085ae6L, 0xff0f6a70L, 0x66063bcaL, 0x11010b5cL,
    0x8f659effL, 0xf862ae69L, 0x616bffd3L, 0x166ccf45L,
    0xa00ae278L, 0xd70dd2eeL, 0x4e048354L, 0x3903b3c2L,
    0xa7672661L, 0xd06016f7L, 0x4969474dL, 0x3e6e77dbL,
    0xaed16a4aL, 0xd9d65adcL, 0x40df0b66L, 0x37d83bf0L,
    0xa9bcae53L, 0xdebb9ec5L, 0x47b2cf7fL, 0x30b5ffe9L,
    0xbdbdf21cL, 0xcabac28aL, 0x53b39330L, 0x24b4a3a6L,
    0xbad03605L, 0xcdd70693L, 0x54de5729L, 0x23d967bfL,
    0xb3667a2eL, 0xc4614ab8L, 0x5d681b02L, 0x2a6f2b94L,
    0xb40bbe37L, 0xc30c8ea1L, 0x5a05df1bL, 0x2d02ef8dL
};

// See: https://blog.csdn.net/ubuntu64fan/article/details/90056041
// See: https://blog.csdn.net/weed_hz/article/details/25132343

/**
 * crc32 standard:
 *   tabsize=256
 *   polynomial = 0xEDB88320L
 */
#define CRC32_POLYNOMIAL  0xEDB88320L
#define CRC32_TABLESIZE   256

static uint32_t crc32_table[256];

void init_crc32_table(void)
{
    uint32_t c;
    uint32_t i, j;

    for (i = 0; i < 256; i++) {
        c = i;
        for (j = 0; j < 8; j++) {
            if (c & 1)
                c = (c >> 1) ^ CRC32_POLYNOMIAL;
            else
                c = c >> 1;
        }
        crc32_table[i] = c;
    }
}

uint32_t crc32_x86(const unsigned char * data, size_t length)
{
    uint32_t crc32 = 0xFFFFFFFF;
    for (size_t i = 0; i < length; i++) {
        crc32 = crc32_table[(crc32 & 0xFF) ^ (*data)] ^ (crc32 >> 8);
        data++;
    }
    crc32 = crc32 ^ 0xFFFFFFFF;
    return crc32;
}

uint32_t crc32c_x86(const unsigned char * data, size_t length)
{
    uint32_t crc32 = 0xFFFFFFFF;
    for (size_t i = 0; i < length; i++) {
        crc32 = CRC32c_Table[(crc32 & 0xFF) ^ (*data)] ^ (crc32 >> 8);
        data++;
    }
    crc32 = crc32 ^ 0xFFFFFFFF;
    return crc32;
}
```

CRC32为例详细解析（菜鸟至老鸟进阶）

https://www.cnblogs.com/masonzhang/p/10261855.html

这篇文章比较详细的介绍了 CRC32 算法的原理.

CRC-知识解析 cyclic redundancy check

           Normal      Reversed    Reciprocal  Reversed reciprocal

CRC-32   0x04C11DB7   0xEDB88320   0xDB710641  0x82608EDB


CRC-32(normal) 标准引用:  ZIP，RAR，IEEE 802 LAN/FDDI，IEEE1394，PPP-FCS

CRC-32c: 0x1EDC6F41 标准引用: SCTP



如何计算CRC32校验和？

[https://www.thinbug.com/q/2587766](https://www.thinbug.com/q/2587766)

[CRC primer, Chapter 7](http://chrisballance.com/wp-content/uploads/2015/10/CRC-Primer.html)

仅使用32位数作为除数，并使用整个流作为被除数。丢掉商并保留其余部分。在邮件结尾处填写余数并且您有CRC32。

计算方法：

```text
         QUOTIENT
        ----------
DIVISOR ) DIVIDEND
                 = REMAINDER
```

1. 取前32位。
2. 转移位
3. 如果32位小于 DIVISOR，请转到步骤2.
4. DIVISOR 的 XOR 32位。转到第2步。
（请注意，流必须可以被32位分割，或者它应该被填充。例如，必须填充8位ANSI流。同样在流的末尾，分割被暂停。）

答案 3 :(得分：5)

对于IEEE802.3，CRC-32。将整个消息视为串行比特流，在消息末尾附加32个零。接下来，必须反转消息的每个字节的位，并对前32位执行1的补码。现在除以CRC-32多项式0x104C11DB7。最后，你必须1对这个除法的32位余数进行补码，对每个剩余的4个字节进行反转。这将成为附加到消息末尾的32位CRC。

这个奇怪程序的原因是第一个以太网实现会一次一个字节地串行化消息，并首先发送每个字节的最低有效位。然后串行比特流经过串行CRC-32移位寄存器计算，该消息在消息完成后被简单地补充并在线路上发送出去。补充消息的前32位的原因是，即使消息全为零，也不会得到全零CRC。

答案 4 :(得分：4)

我花了一些时间试图揭开这个问题的答案，我终于在今天发表了关于CRC-32的教程：[CRC-32 hash tutorial - AutoHotkey Community](https://autohotkey.com/boards/viewtopic.php?f=7&t=35671)

在这个例子中，我演示了如何计算ASCII字符串 'abc' 的 CRC-32 哈希值：

```text
calculate the CRC-32 hash for the ASCII string 'abc':

inputs:
dividend: binary for 'abc': 0b011000010110001001100011 = 0x616263

polynomial: 0b100000100110000010001110110110111 = 0x104C11DB7

dividend "abc" byte:
011000010110001001100011

reverse bits in each byte:
100001100100011011000110

append 32 0 bits:
10000110010001101100011000000000000000000000000000000000

XOR the first 4 bytes with 0xFFFFFFFF:
01111001101110010011100111111111000000000000000000000000

'CRC division':

01111001101110010011100111111111000000000000000000000000
 100000100110000010001110110110111
 ---------------------------------
  111000100010010111111010010010110
  100000100110000010001110110110111
  ---------------------------------
   110000001000101011101001001000010
   100000100110000010001110110110111
   ---------------------------------
    100001011101010011001111111101010
    100000100110000010001110110110111
    ---------------------------------
         111101101000100000100101110100000
         100000100110000010001110110110111
         ---------------------------------
          111010011101000101010110000101110
          100000100110000010001110110110111
          ---------------------------------
           110101110110001110110001100110010
           100000100110000010001110110110111
           ---------------------------------
            101010100000011001111110100001010
            100000100110000010001110110110111
            ---------------------------------
              101000011001101111000001011110100
              100000100110000010001110110110111
              ---------------------------------
                100011111110110100111110100001100
                100000100110000010001110110110111
                ---------------------------------
                    110110001101101100000101110110000
                    100000100110000010001110110110111
                    ---------------------------------
                     101101010111011100010110000001110
                     100000100110000010001110110110111
                     ---------------------------------
                       110111000101111001100011011100100
                       100000100110000010001110110110111
                       ---------------------------------
                        10111100011111011101101101010011

remainder: 0b10111100011111011101101101010011 = 0xBC7DDB53

XOR the remainder with 0xFFFFFFFF:
0b01000011100000100010010010101100 = 0x438224AC

reverse bits:
0b00110101001001000100000111000010 = 0x352441C2

thus the CRC-32 hash for the ASCII string 'abc' is 0x352441C2
```
