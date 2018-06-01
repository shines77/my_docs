
# String Matching (字符串匹配算法)

* `EXACT STRING MATCHING ALGORITHMS (Animation in Java)`

    [http://www-igm.univ-mlv.fr/~lecroq/string/]()

    经典的字符串算法介绍。

* `字符串匹配算法的分析`

    [https://www.cnblogs.com/adinosaur/p/6002978.html]()

    1. 朴素算法(Brute Force算法)
    2. Rabin-Karp算法
    3. 有限状态自动机
    4. KMP算法

* `几种常见的字符串匹配算法`

    [https://blog.csdn.net/qingkongyeyue/article/details/53464437]()

    1. 朴素算法(Brute Force算法)
    2. Rabin-Karp算法
    3. KMP算法
    4. Boyer-Moore算法
    5. Sunday算法

* `字符串匹配算法KMP详细解释——深入理解`

    [https://blog.csdn.net/FX677588/article/details/53406327]()

* `AC 自动机的实现`

    [http://nark.cc/p/?p=1453]()

    我们已使用 terark.com 进行商业化运营, 产品包括: 数据库, 存储引擎, 多模匹配 多正则匹配, 输入纠错等。

* `AC 自动机算法详解`

    [https://www.cnblogs.com/UnGeek/p/5734844.html]()

    AC自动机：Aho-Corasick automation，该算法在1975年产生于贝尔实验室，是著名的多模匹配算法之一。

* `深入理解 Aho-Corasick 自动机算法`

    [https://blog.csdn.net/lemon_tree12138/article/details/49335051]()

    还不错的 AC 自动机算法介绍的文章。

* `多模式串匹配 之 AC自动机算法（Aho-Corasick算法）简介`

    [http://lzw.me/a/1274.html]()

    AC 自动机算法全称 Aho-Corasick 算法，是一种字符串多模式匹配算法。该算法在1975年产生于贝尔实验室，是著名的多模匹配算法之一。

* `如何更好的理解和掌握 KMP 算法?`

    [https://www.zhihu.com/question/21923021]()


    回答者：陈硕

    为什么要执着于 KMP 呢？
    
    glibc 2.9 之后的 strstr() 在一定情况下会用高效的 [Two Way algorithm](https://www-igm.univ-mlv.fr/%7Elecroq/string/node26.html)，之前的版本是普通的二重循环查找，因此用不着自己写。
    
    而且 glibc 的作者一度也写错过：<br/>
    https://sourceware.org/bugzilla/show_bug.cgi?id=12092 。
    
    ps: strstr() 还不止一次出过 bug：<br/>
    https://sourceware.org/bugzilla/show_bug.cgi?id=12100<br/>
    https://sourceware.org/bugzilla/show_bug.cgi?id=14602<br/>
    等等。
    
    其他：[c - What is the fastest substring search algorithm?](https://stackoverflow.com/questions/3183582/what-is-the-fastest-substring-search-algorithm)

* `SSE 4.2 带来的优化`

    [https://www.zzsec.org/2013/08/using-sse_4.2/]()

    PCMPESTRI，PCMPESTRM，PCMPISTRI 和 PCMPISTRM 指令。

    * [http://www.strchr.com/strcmp_and_strlen_using_sse_4.2]()
    * [http://www.intel.com/content/dam/www/public/us/en/documents/manuals/64-ia-32-architectures-software-developer-vol-2b-manual.pdf]()

* `Implementing strcmp, strlen, and strstr using SSE 4.2 instructions`

    [https://www.strchr.com/strcmp_and_strlen_using_sse_4.2]()

    strcmp_sse42(), strlen_sse42(), strstr_sse42() 。

* `PCMPxSTRx 命令の使い方`

    [http://www.nminoru.jp/~nminoru/programming/pcmpxstrx.html]()

    一篇介绍 PCMPxSTRx 指令详细用法的日文文章，还不错，我已经翻译成中文了。

* `strstr() 与 Intel SSE 指令集优化`

    [https://www.jianshu.com/p/d718c1ea5f22]()

    还不错，做个各个版本的性能比较。

