# HashMap 相关文章

* `HashMap 源码详细分析(JDK1.8)`

    [https://segmentfault.com/a/1190000012926722](https://segmentfault.com/a/1190000012926722)

    ```
    HashMap 底层是基于散列算法实现，散列算法分为散列再探测和拉链式。HashMap 则使用了拉链式的散列算法，并在 JDK 1.8 中引入了红黑树优化过长的链表。

    HashMap 基本操作就是对拉链式散列算法基本操作的一层包装。不同的地方在于 JDK 1.8 中引入了红黑树，底层数据结构由数组 + 链表变为了数组 + 链表 + 红黑树，不过本质并未变。

    初始容量(initialCapacity)、负载因子(loadFactor)、阈值(threshold)。

    该文分析得还算透彻，值得一读。
    ```

* `HashMap实现原理分析`

    [https://blog.csdn.net/vking_wang/article/details/14166593](https://blog.csdn.net/vking_wang/article/details/14166593)

    ```
    这篇文章介绍的大概是 JDK 1.7 以前的 HashMap 版本。

    HashMap里面实现一个静态内部类Entry，其重要的属性有 key , value, next，从属性key,value我们就能很明显的看出来Entry就是HashMap键值对实现的一个基础bean，我们上面说到HashMap的基础就是一个线性数组，这个数组就是Entry[]，Map里面的内容都保存在Entry[]里面。

* `上古时代 Objective-C 中哈希表的实现`

    [https://segmentfault.com/a/1190000005075494](https://segmentfault.com/a/1190000005075494)

    ```
    文章介绍 Mac OS 上的 Objective-C 哈希表，也就是 NXHashTable ：

    NXHashTable 的实现
    NXHashTable 的性能分析
    NXHashTable 的作用

    NXHashTable 的实现有着将近 30 年的历史，不过仍然作为重要的底层数据结构存储整个应用中的类。
    ```

* `知乎: HashMap底层实现原理（上）`

    [https://zhuanlan.zhihu.com/p/28501879](https://zhuanlan.zhihu.com/p/28501879)

    ```
    链表的节点数超过 8 时，转换为红黑树。
    ```

* `知乎: HashMap底层实现原理（下）`

    [https://zhuanlan.zhihu.com/p/28587782](https://zhuanlan.zhihu.com/p/28587782)


    ```
    接上文。
    ```

* `知乎: Hash时取模一定要模质数吗？`

    [https://www.zhihu.com/question/20806796](https://www.zhihu.com/question/20806796)

    ```
    作者：十一太保念技校

    首先来说假如关键字是随机分布的，那么无所谓一定要模质数。但在实际中往往关键字有某种规律，例如大量的等差数列，那么公差和模数不互质的时候发生碰撞的概率会变大，而用质数就可以很大程度上回避这个问题。质数并不是唯一的准则，具体可以参考以下网站。
    
    good hash table primes

    作者：Yining

    举个例子，对于除法哈希表（Division method）h(k)=k mod m注意二进制数对取余就是该二进制数最后r位数。这样一来，Hash函数就和键值（用二进制表示）的前几位数无关了，这样我们就没有完全用到键值的信息，这种选择m的方法是不好的。（然而，这种方法却可以使得我们简化截取二进制数中间某几位的操作，只要利用mod运算和无关位取0运算即可）所以好的方法就是用质数来表示m，使得这些质数，不太接近2的幂或者10的幂。
    ```

* `Good hash table primes`

    [http://planetmath.org/goodhashtableprimes](http://planetmath.org/goodhashtableprimes)

    ```
    In the course of designing a good hashing configuration, it is helpful to have a list of prime numbers for the hash table size.

    The following is such a list. It has the properties that:

    1. each number in the list is prime
    2. each number is slightly less than twice the size of the previous
    3. each number is as far as possible from the nearest two powers of two

    Using primes for hash tables is a good idea because it minimizes clustering in the hashed table. Item (2) is nice because it is convenient for growing a hash table in the face of expanding data. Item (3) has, allegedly, been shown to yield especially good results in practice.

    And here is the list:

    lwr	upr	% err	        prime
    25	26	10.416667	53
    26	27	1.041667	97
    27	28	0.520833	193
    28	29	1.302083	389
    29	210	0.130208	769
    210	211	0.455729	1543
    211	212	0.227865	3079
    212	213	0.113932	6151
    213	214	0.008138	12289
    214	215	0.069173	24593
    215	216	0.010173	49157
    216	217	0.013224	98317
    217	218	0.002543	196613
    218	219	0.006358	393241
    219	220	0.000127	786433
    220	221	0.000318	1572869
    221	222	0.000350	3145739
    222	223	0.000207	6291469
    223	224	0.000040	12582917
    224	225	0.000075	25165843
    225	226	0.000010	50331653
    226	227	0.000023	100663319
    227	228	0.000009	201326611
    228	229	0.000001	402653189
    229	230	0.000011	805306457
    230	231	0.000000	1610612741

    The columns are, in order, the lower bounding power of two, the upper bounding power of two, the relative deviation (in percent) of the prime number from the optimal middle of the first two, and finally the prime itself.

    ```

* `为什么hash table的大小最好取一个不接近 2^p 的素数`

    [http://blog.51cto.com/thuhak/1352903](http://blog.51cto.com/thuhak/1352903)

    ```
    1. m 不能取一个 2^p 的数。算法导论是这样解释的：

      这是因为对一个数除以 2^p 取余数相当于只取这个数的最低的 p 位，高于 p 位的信息就被丢弃了。

    2. m 不能取一个接近 2^p 或 10^p 的数。

    3. m 的值为什么要取一个素数？
    ```
