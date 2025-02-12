# 无需重新排序的开放地址哈希表的最优边界

## 出处

原文：[[Optimal Bounds for Open Addressing Without Reordering]](https://ar5iv.org/html/2501.02305)

PDF版：[https://arxiv.org/pdf/2501.02305](https://arxiv.org/pdf/2501.02305)

## 作者

1. [Martín Farach-Colton](mailto:martin@farach-colton.com) (马丁·法拉赫 - 科尔顿)，纽约大学。

2. [Andrew Krapivin](mailto:andrew@krapivin.net) (安德鲁·克拉皮文)，剑桥大学。

3. [William Kuszmaul](mailto:kuszmaul@cmu.edu) (威廉·库兹莫尔)，卡内基梅隆大学。

## 摘要

在本文中，我们重新审视数据结构中一个极为基础的问题：将元素插入开放地址哈希表，确保后续检索时探测次数尽可能少。我们发现，即便不随时间重排元素，构建的哈希表在期望搜索复杂度（包括均摊和最坏情况）上，也能远超此前预期。在此过程中，我们证伪了姚期智在其开创性论文《均匀哈希是最优的》中留下的核心猜想。我们所有结果都给出了对应的下界。

## 1. 引言

在本文中，我们重新审视数据结构中一个最简单的问题：将元素插入开放地址哈希表，以便日后能以尽可能少的探测次数检索到这些元素。我们表明，即使不随时间重新排列元素，也有可能构建出一种哈希表，其期望探测复杂度（包括均摊和最坏情况）比之前认为的要好得多。在此过程中，我们反驳了姚（Yao）在其开创性论文《均匀哈希是最优的》[21] 中留下的核心猜想。

### 背景

考虑构建一个不重排的开放地址哈希表这一基本问题。将一系列键值对 $x_{1},x_{2},\ldots,x_{(1-\delta)n}$ 依次插入大小为 $n$ 的数组中。每个 $x_{i}$ 都有一个探测序列 $h_{1}(x_{i}),h_{2}(x_{i}),\ldots\in[n]^{\infty}$，该序列是从某个分布 $\mathcal{P}$ 中独立抽取的。为插入元素 $x_{i}$，插入算法 $\mathcal{A}$ 必须从尚未被占用的位置中选择 $h_{j}(x)$ 来放置该元素。需要注意的是，插入操作不能重新排列（即移动）之前插入的元素，所以插入操作的唯一任务就是选择一个未占用的槽位。哈希表的完整描述由对 $(\mathcal{P},\mathcal{A})$ 给出。

如果 $x_{i}$ 被放置在位置 $h_{j}(x_{i})$，那么 $x_{i}$ 的探测复杂度为 $j$。这意味着查询时通过对位置 $h_{1}(x),\ldots h_{j}(x)$ 进行 $j$ 次探测就能找到 $x_{i}$。我们的目标是设计哈希表 $(\mathcal{P},\mathcal{A})$，使得均摊期望探测复杂度最小化，即所有键值对 $x_{1},x_{2},\ldots,x_{(1-\delta)n}$ 的平均探测复杂度的期望值最小。

这个问题的经典解决方案是使用均匀探测法 [13]：每个键的探测序列是 $\{1,2,\ldots,n\}$ 的随机排列，每次插入 $x_{i}$ 时，贪心算法会从其探测序列中选择第一个未被占用的位置。通过简单计算可知，随机探测的均摊期望探测复杂度为 $\Theta(\log\delta^{-1})$ 。

1972 年，厄尔曼（Ullman）猜想 [19]，对于所有贪心算法（即每个元素都使用其探测序列中第一个未被占用位置的算法）而言，$\Theta(\log\delta^{-1})$ 的均摊期望探测复杂度是最优的。在 1985 年姚（Yao）发表的一篇著名论文《均匀哈希是最优的》[21] 中，这个猜想才得以证明。

突破姚（Yao）提出的下界的经典方法是放宽问题条件，允许插入算法进行重排操作，即在元素插入后移动它们的位置。在这种宽松的设定下，即使哈希表完全填满，也能实现 $O(1)$ 的均摊期望探测复杂度 [6, 10, 17] 。但尚不清楚这种放宽是否必要。非贪心算法能否在不重排的情况下，实现 $o(\log\delta^{-1})$ 的均摊期望探测复杂度呢？或者说，为了实现较小的均摊探测复杂度，重排操作是否在根本上是必要的呢？

> 问题 1
>
> 开放地址哈希表在插入元素后不进行重排，能否实现 $o(\log\delta^{-1})$ 的均摊期望探测复杂度？

一个与之密切相关的问题是最小化最坏情况期望探测复杂度。最坏情况期望探测复杂度的界限必须单独适用于每一次插入操作，即使是在哈希表非常满时进行的插入操作。均匀探测法的最坏情况期望探测复杂度为 $O(\delta^{-1})$ 。然而，在不使用重排操作的情况下，这个界限是否是渐近最优的，仍然是一个悬而未决的问题。

> 问题 2
>
> 开放地址哈希表在不重排的情况下，能否实现 $o(\delta^{-1})$ 的最坏情况期望探测复杂度？

第二个问题相当著名 [19, 21, 15, 8, 16, 14]，即使对于贪心开放地址哈希表，它也仍然没有答案。1985 年，姚（Yao）猜想 [21]，在这种情况下均匀探测法应该是近乎最优的，即任何贪心开放地址哈希表的最坏情况期望探测复杂度至少为 $(1 - o(1))\delta^{-1}$ 。尽管这个猜想表述简单，但从未得到解决。

### 本文：无重排开放地址哈希表的严格界限

在第 2 节中，我们给出了一种哈希表，对上述两个问题都给出了肯定的答案。具体来说，我们展示了如何在不使用重排操作的开放地址哈希表中，实现期望探测复杂度的均摊界限为 $O(1)$，最坏情况界限为 $O(\log\delta^{-1})$ 。

#### 定理 1

设 $n\in\mathbb{N}$ 和 $\delta\in(0,1)$ 为参数，满足 $\delta>O(1/n)$ 且 $\delta^{-1}$ 是 2 的幂次方。可以构造一个开放地址哈希表，在大小为 $n$ 的数组中支持 $n-\lfloor\delta n\rfloor$ 次插入操作，插入后不重排元素，并且均摊期望探测复杂度为 $O(1)$，最坏情况期望探测复杂度为 $O(\log\delta^{-1})$，最坏情况期望插入时间为 $O(\log\delta^{-1})$ 。

我们将插入策略称为弹性哈希（elastic hashing），因为哈希表在确定最终使用的位置之前，常常会在探测序列中深入探测更远的位置。也就是说，在决定将键 $x$ 放入哪个槽位 $h_{i}(x)$ 时，算法会先检查许多满足 $j>i$ 的槽位 $h_{j}(x)$ 。这种非贪心行为至关重要，因为这是在不重排的情况下，有望避开姚（Yao）下界 [21] 的唯一方法。

我们得到的均摊期望探测复杂度为 $O(1)$ ，这显然是最优的。但最坏情况期望探测复杂度为 $O(\log\delta^{-1})$ 这个界限呢？我们证明这个界限也是最优的：任何不使用重排操作的开放地址哈希表，其最坏情况期望探测复杂度至少为 $\Omega(\log\delta^{-1})$ 。

接下来，在第 3 节中，我们将注意力转向贪心开放地址哈希表。回想一下，在这种情况下，问题 1 已经得到解决——几十年来人们都知道均匀探测法在渐近意义上是最优的 [21] 。另一方面，问题 2 仍然悬而未决——正是在这种情况下，姚（Yao）猜想均匀探测法是最优的 [19, 21, 15, 8, 16, 14] 。我们的第二个结果是一种简单的贪心开放地址策略，称为漏斗哈希（funnel hashing），它实现了 $O(\log^{2}\delta^{-1})$ 的最坏情况期望探测复杂度：

#### 定理 2

设 $n\in\mathbb{N}$ 和 $\delta\in(0,1)$ 为参数，满足 $\delta>O(1/n^{o(1)})$ 。存在一种贪心开放地址策略，在大小为 $n$ 的数组中支持 $n-\lfloor\delta n\rfloor$ 次插入操作，并且最坏情况期望探测复杂度（和插入时间）为 $O(\log^{2}\delta^{-1})$ 。此外，该策略保证，在概率为 $1 - 1/{poly}(n)$ 的情况下，所有插入操作的最坏情况探测复杂度为 $O(\log^{2}\delta^{-1}+\log\log n)$ 。最后，均摊期望探测复杂度为 $O(\log\delta^{-1})$ 。

$O(\log^{2}\delta^{-1})=o(\delta^{-1})$ 的最坏情况期望探测复杂度渐近小于姚（Yao）猜想的最优界限 $\Theta(\delta^{-1})$ 。这意味着姚（Yao）的猜想是错误的，并且在某种意义上，即使在贪心开放地址策略中，均匀探测法也不是最优的。

尽管定理 2 中的界限形式有些不寻常，但它们实际上是最优的。对于最坏情况期望探测复杂度，我们证明了任何贪心开放地址方案都有一个匹配的下界 $\Omega(\log^{2}\delta^{-1})$ 。对于高概率最坏情况探测复杂度，我们证明了任何不执行重排的开放地址算法都有一个匹配的下界 $\Omega(\log^{2}\delta^{-1}+\log\log n)$ 。或许令人惊讶的是，这个第二个下界甚至适用于非贪心策略。

我们用于证明定理 2 的漏斗哈希表的基本结构非常简单。在本文的初始版本发布之后，作者还了解到在不同场景下有几种其他哈希表也采用了类似的高层思想 [7, 9] 。多级自适应哈希 [7] （但仅在低负载因子下）使用了类似的结构，以获得一个具有 $O(\log\log n)$ 级别的哈希表，该哈希表在查询时支持高并行性——这个思想随后也被应用于竞争解决方案的设计 [4] 。过滤哈希 [9] 将该结构应用于高负载因子，作为 $d$ 元布谷鸟哈希的替代方案。与标准 $d$ 元布谷鸟哈希的已知分析不同，过滤哈希可以用常数时间的多项式哈希函数实现。过滤哈希实现了与漏斗哈希相同的 $O(\log^{2}\delta^{-1})$ 类型的界限，实际上，漏斗哈希可以看作是过滤哈希的一种变体，经过修改后成为贪心开放地址的一个实例，并具有最优的最坏情况探测复杂度，从而证明了定理 2 并反驳了姚（Yao）的猜想 [21] 。

### 其他问题历史和相关工作

在引言的结尾，我们简要讨论一下相关工作，以及本文所研究问题和模型的历史。

研究均摊期望探测复杂度的想法似乎最早由克努特（Knuth）在 1963 年关于线性探测的论文 [11] 中提出。克努特观察到，当线性探测哈希表的填充率为 $1-\delta$ 时，尽管期望插入时间为 $O(\delta^{-2})$ ，但均摊期望探测复杂度为 $O(\delta^{-1})$ 。克努特后来提出了厄尔曼（Ullman）猜想的一个较弱版本 [12] ，即均匀探测法在一类被称为单哈希策略的受限贪心策略中是最优的。这个较弱的猜想随后被阿伊泰（Ajtai）证明 [1] ，他的技术最终成为姚（Yao）证明完整猜想 [21] 的基础。如前所述，姚（Yao）猜想应该可以得到一个更强的结果，即任何贪心开放地址哈希表的最坏情况期望探测复杂度为 $\Omega(\delta^{-1})$ 。这个猜想一直悬而未决 [15, 8, 16, 14] ，直到本文通过定理 2 对其进行了反驳。

尽管本文没有讨论键值对，但开放地址法的大多数应用都会为每个键关联一个值 [13] 。在这些场景中，查询的任务不一定是确定键是否存在（通常已经知道键存在），而是恢复相应的值。这种区别很重要，因为探测复杂度和均摊探测复杂度这两个概念仅适用于存在的键。特别是，最小化均摊期望探测复杂度对应于最小化查询随机一个存在元素的期望时间。对于否定查询（即查询不存在的元素），则没有类似的概念——在查询不存在的元素时，查询任意一个元素与查询随机一个元素之间没有有趣的区别。

另一方面，对于最坏情况期望探测复杂度，在某些情况下可以将结果扩展到否定查询。特别是对于贪心算法，否定查询时间与插入时间相同（两者在遇到空闲槽时都会停止） [13] 。因此，定理 2 中的保证意味着否定查询的期望时间界限为 $O(\log^{2}\delta^{-1})$ 。

还可以将无重排开放地址法的研究扩展到支持在无限时间范围内进行插入和删除操作的场景 [18, 3, 2] 。在这种场景下，即使是像线性探测 [18] 和均匀探测 [3] 这样非常基本的方案也难以分析——目前还不清楚这两种方案是否能实现以 $\delta^{-1}$ 为函数的期望插入时间、探测复杂度或均摊探测复杂度。然而，已知在这种场景下最优的均摊期望探测复杂度为 $\delta^{-\Omega(1)}$ （见 [3] 中的定理 3），这意味着像定理 1 和定理 2 这样的结果是不可能的。

## 2. 弹性哈希 (Elastic Hashing)

在本节中，我们构造弹性哈希，这是一种开放地址哈希表（不进行重排），能够实现 $O(1)$ 的均摊期望探测复杂度和 $O(\log\delta^{-1})$ 的最坏情况期望探测复杂度。

我们的构造将使用一个特定的单射 $\phi:\mathbb{Z}^{+}\times\mathbb{Z}^{+}\to\mathbb{Z}^{+}$ 。

### 引理 1

存在一个单射 $\phi:\mathbb{Z}^{+}\times\mathbb{Z}^{+}\to\mathbb{Z}^{+}$ ，使得 $\phi(i,j)\leq O(i\cdot j^{2})$ 。

#### 证明

取 $i$ 的二进制表示 $a_{1}\circ a_{2}\circ\cdots\circ a_{p}$ 和 $j$ 的二进制表示 $b_{1}\circ b_{2}\circ\cdots\circ b_{q}$ （这里，$a_{1}$ 和 $b_{1}$ 分别是 $i$ 和 $j$ 的最高位），构造 $\phi(i,j)$ 的二进制表示为：

$$1\circ b_{1}\circ 1\circ b_{2}\circ 1\circ b_{3}\circ\cdots\circ 1\circ b_{1}\circ 0\circ a_{1}\circ a_{2}\circ\ldots\circ a_{p}$$

同样，这些数位是从最高位到最低位读取的。根据设计，映射 $\phi(i,j)$ 是一个单射，因为可以从 $\phi(i,j)$ 的二进制表示中直接恢复 $i$ 和 $j$ 。另一方面，

$$\log_{2}\phi(i,j)\leq\log_{2}i + 2\log_{2}j + O(1)$$

这意味着 $\phi(i,j)\leq O(i\cdot j^{2})$ ，证毕。

### 算法

现在我们描述插入算法。将大小为 $n$ 的数组 $A$ 划分为不相交的数组 $A_{1},A_{2},\ldots,A_{\lceil\log n\rceil}$ ，满足 $|A_{i + 1}| = |A_{i}|/2\pm 1$ 。

我们将模拟一个二维探测序列 $\{h_{i,j}\}$ ，其中探测 $h_{i,j}(x)$ 是数组 $A_{i}$ 中的一个随机槽位。具体来说，我们通过定义

$$h_{\phi(i,j)}(x):=h_{i,j}(x)$$

将二维序列 $\{h_{i,j}\}$ 的元素映射到一维序列 $\{h_{i}\}$ 中。这意味着放置在槽位 $h_{i,j}(x)$ 中的元素 $x$ 的探测复杂度为 $O(i\cdot j^{2})$ 。

我们将 $n-\lfloor\delta n\rfloor$ 次插入操作划分为若干批次 $\mathcal{B}_{0},\mathcal{B}_{1},\mathcal{B}_{2},\ldots$ 。批次 $\mathcal{B}_{0}$ 将数组 $A_{1}$ 填充到有 $\lceil 0.75|A_{1}|\rceil$ 个元素，每个元素 $x$ 都使用探测序列 $h_{1,1}(x),h_{1,2}(x),h_{1,3}(x),\ldots$ 中的第一个可用槽位进行插入。对于 $i\geq 1$ ，批次 $\mathcal{B}_{i}$ 包含：

$$|A_{i}|-\lfloor\delta|A_{i}|/2\rfloor-\lceil 0.75\cdot|A_{i}|\rceil+\lceil 0.75\cdot|A_{i + 1}|\rceil$$

次插入操作，这些插入操作都在数组 $A_{i}$ 和 $A_{i + 1}$ 中进行（最后一个批次可能未完成，因为插入操作已用完）。对于 $i\geq 0$ ，在批次 $\mathcal{B}_{i}$ 结束时的保证是，每个满足 $j\in\{1,\ldots,i\}$ 的 $A_{j}$ 恰好包含 $|A_{j}|-\lfloor\delta|A_{j}|/2\rfloor$ 个元素，并且 $A_{i + 1}$ 恰好包含 $\lceil 0.75\cdot|A_{i + 1}|\rceil$ 个元素。注意，这个保证决定了批次大小由式(1)给出。此外，由于插入操作的总数为 $n - \lfloor\delta n\rfloor$，并且每个批次 $\mathcal{B}_{i}$ 在整个数组 $A$ 中最多留下 $O(n/2^{i}) + \delta n/2$ 个剩余空闲槽位，所以插入序列保证在 $O(\log\delta^{-1})$ 个批次内完成。

设 $c$ 为一个参数，我们稍后将其设置为一个较大的正常数，并定义函数：

$$f(\epsilon)=c\cdot\min(\log^{2}\epsilon^{-1},\log\delta^{-1})$$

现在我们描述在批次 $\mathcal{B}_{i}$（$i\geq1$）期间如何插入元素 $x$。假设在插入时，$A_{i}$ 的填充率为 $1 - \epsilon_{1}$，$A_{i + 1}$ 的填充率为 $1 - \epsilon_{2}$ 。有三种情况：

1. 如果 $\epsilon_{1}>\delta/2$ 且 $\epsilon_{2}>0.25$，那么 $x$ 可以放入 $A_{i}$ 或 $A_{i + 1}$，放置方式如下：如果 $A_{i}$ 中的位置 $h_{i,1}(x),h_{i,2}(x),\ldots,h_{i,f(\epsilon_{1})}(x)$ 中有任何一个空闲，则将 $x$ 放置在第一个这样的空闲槽位中；否则，将 $x$ 放置在序列 $h_{i + 1,1}(x),h_{i + 1,2}(x),h_{i + 1,3}(x),\ldots$ 的第一个空闲槽位中。
2. 如果 $\epsilon_{1}\leq\delta/2$，那么$x$必须放置在 $A_{i + 1}$ 中，并且将 $x$ 放置在序列 $h_{i + 1,1}(x),h_{i + 1,2}(x),h_{i + 1,3}(x),\ldots$ 的第一个空闲槽位中。
3. 最后，如果 $\epsilon_{2}\leq0.25$，那么 $x$ 必须放置在 $A_{i}$ 中，并且将 $x$ 放置在序列 $h_{i,1}(x),h_{i,2}(x),h_{i,3}(x),\ldots$ 的第一个空闲槽位中。

我们将最后一种情况称为昂贵情况，因为 $x$ 是插入到可能非常满的数组 $A_{i}$ 中，并且使用的是均匀探测法。不过，我们稍后会看到，这种情况非常罕见：在批次 $\mathcal{B}_{i}$ 期间，这种情况发生的概率为 $1 - O(1/|A_{i}|^{2})$ 。

需要注意的是，情况 2 和情况 3 是不相交的（在给定的批次中，这两种情况中只有一种可能发生），因为一旦 $\epsilon_{1}\leq\delta/2$且$\epsilon_{2}\leq0.25$ 同时成立，该批次就结束了。

### 绕过优惠券收集器的瓶颈

在深入分析之前，从高层次上理解我们的算法如何绕过均匀探测所面临的“优惠券收集器”瓶颈是很有帮助的。在均匀探测中，每次探测可以看作是对一个随机优惠券（即槽位）的采样；标准的优惠券收集器下界表明，如果要收集 $(1 - \delta)$ 比例的优惠券，至少需要进行 $\Omega(n\log\delta^{-1})$ 次探测。这就使得均匀探测（或任何类似均匀探测的方法）无法实现比 $O(\log\delta^{-1})$ 更好的均摊期望探测复杂度。

我们算法的一个关键特性是它将每个键的插入探测复杂度（即插入键时进行的探测次数）与其搜索探测复杂度（即找到键所需的探测次数）分离开来。当然，后者通常就是我们简单称为探测复杂度的量，但为了在本节中避免歧义，我们有时会称其为搜索探测复杂度。

插入算法通常会在探测序列中比最终使用的位置探测更远的位置。乍一看，这样的插入探测可能似乎没有用处，但正如我们将看到的，它们是避免优惠券收集器瓶颈的关键——结果是大多数优惠券只对插入探测复杂度有贡献，而对搜索探测复杂度没有贡献。

为了看到这种分离的实际效果，考虑在批次 $\mathcal{B}_{1}$ 中的一次插入操作，假设此时 $A_{1}$ 的填充率为 $(1 - 2\delta^{-1})$，$A_{2}$ 的填充率为 $0.6$。这次插入对 $A_{1}$ 进行了 $\Theta(f(\delta^{-1})) = \Theta(\log\delta^{-1})$ 次探测，但很可能所有这些探测都失败了（每次探测成功的概率仅为 $O(\delta)$）。然后插入操作在 $A_{2}$ 中寻找空闲槽位，并且很可能最终使用了形式为 $h_{\phi(2,j)}$ 的位置，其中 $j = O(1)$，从而导致搜索探测复杂度为 $O(\phi(2,j)) = O(1)$。所以，在这个例子中，即使插入探测复杂度为 $\Theta(\log\delta^{-1})$，搜索探测复杂度却是 $O(1)$ 。

优惠券收集器瓶颈也是难以实现最坏情况期望插入（和搜索）界限的原因。我们知道必须收集 $\Theta(n\log\delta)$ 个总优惠券，直观地说，最后插入的元素（即在高负载因子下进行的插入操作）必须完成大部分收集工作。毕竟，在低负载因子下的插入操作怎么能有效地利用超过几个优惠券呢？这就是导致像均匀探测这样的算法具有 $O(\delta^{-1})$ 的最坏情况期望插入时间的原因。

我们的算法也绕过了这个瓶颈：即使总共收集了 $\Theta(n\log\delta^{-1})$ 个优惠券，但没有一次插入的期望贡献超过 $O(\log\delta^{-1})$。这意味着即使是在低负载因子下进行的插入操作，也需要能够“有效地”利用 $\Theta(\log\delta^{-1})$ 次探测/优惠券。这是如何实现的呢？关键在于，一定比例的插入元素 $x$ 有以下经历：当 $x$（在某个批次 $\mathcal{B}_{i}$ 中）插入时，数组 $A_{i}$ 已经几乎满了（所以$x$可以在该数组中有效地采样 $O(\log\delta^{-1})$ 次探测/优惠券），但下一个数组 $A_{i + 1}$ 不是很满（所以如果在 $A_{i}$ 中的 $O(\log\delta^{-1})$ 次探测/优惠券都没有成功，$x$ 很可能可以放入 $A_{i + 1}$ ）。这就是算法能够将优惠券收集器（几乎均匀地！）分布在 $\Theta(n)$ 次操作中的方式。

### 算法分析

我们首先分析给定批次中包含昂贵情况插入操作的概率。

#### 引理 2

在批次 $\mathcal{B}_{i}$ 中，没有插入操作处于昂贵情况（即情况3）的概率至少为 $1 - O(1/|A_{i}|^{2})$ 。

#### 证明

设 $m$ 表示数组 $A_{i}$ 的大小。我们可以假设 $m = \omega(1)$，否则引理显然成立。对于 $j\in\{2,3,\ldots,\lceil\log\delta^{-1}\rceil\}$，设 $T_{j}$ 是批次 $\mathcal{B}_{i}$ 中的一个时间窗口，在这个窗口内，$A_{i}$ 的空闲槽位数从 $\lfloor m/2^{j}\rfloor$ 变为 $\max(\lfloor m/2^{j + 1}\rfloor,\lfloor\delta m/2\rfloor)$ 。

在 $T_{j}$ 期间的每次插入操作都保证处于情况1或情况3，所以插入操作在 $A_{i}$ 中至少进行 $f(2^{-j})$ 次探测尝试，每次尝试成功的概率至少为 $2^{-(j + 1)}$ 。如果 $f(2^{-j})>100\cdot 2^{j}$，那么在时间窗口 $T_{j}$ 内的每次插入操作放置在数组 $A_{i}$ 中的概率至少为 $1 - (1 - 1/2^{j + 1})^{100\cdot 2^{j}}>0.99$。否则，如果 $f(2^{-j})<100\cdot 2^{j}$，那么插入操作放置在数组 $A_{i}$ 中的概率为 $\Theta(f(2^{-j})/2^{j})$。因此，一般来说，$T_{j}$ 中的每次插入操作使用数组 $A_{i}$ 的概率至少为：

$$\min(0.99,\Theta(f(2^{-j})/2^{j}))$$

由此可得：

$$\mathbb{E}[|T_{j}|]\leq\frac{m/2^{j + 1}}{\min(0.99,\Theta(f(2^{-j})/2^{j}))}+O(1)\leq\Theta\left(\frac{m}{f(2^{-j})}\right)+1.02\cdot m/2^{j + 1}+O(1)$$

由于式 (2) 对 $T_{i}$ 中的每次插入操作都独立于之前插入操作的行为成立，我们可以应用切尔诺夫界得出，概率至少为 $1 - 1/m^{3}$ 时，

$$|T_{j}|\leq\mathbb{E}[|T_{j}|]+O(\sqrt{m\log m})\leq O\left(\frac{m}{f(2^{-j})}\right)+1.02\cdot m/2^{j + 1}+O(\sqrt{m\log m})$$

概率至少为 $1 - O(1/m^{2})$ 时，式 (3) 对每个窗口 $T_{j}$ 都成立。

因此，将 $c$ 视为一个参数（而不是一个常数），我们有：

$$\begin{align*}\sum_{j}|T_{j}|&\leq\sum_{j = 2}^{\lceil\log\delta^{-1}\rceil}\left(O\left(\frac{m}{f(2^{-j})}\right)+1.02\cdot m/2^{j + 1}+O(\sqrt{m\log m})\right)\\&\leq 0.26\cdot m+m\cdot O\left(\sum_{j = 2}^{\lceil\log\delta^{-1}\rceil}\frac{1}{f(2^{-j})}\right)+O(1)\\&=0.26\cdot m+m\cdot O\left(\sum_{j = 2}^{\lceil\log\delta^{-1}\rceil}\frac{1}{c\min(j^{2},\log\delta^{-1})}\right)+O(1)\\&\leq 0.26\cdot m+\frac{m}{c}\cdot O\left(\sum_{j = 2}^{\infty}\frac{1}{j^{2}}+\sum_{j = 2}^{\lceil\log\delta^{-1}\rceil}\frac{1}{\log\delta^{-1}}\right)+O(1)\\&\leq 0.26\cdot m+\frac{m}{c}\cdot O(1)+O(1)\end{align*}$$

如果我们将 $c$ 设置为一个足够大的正常数，那么可得 $\sum_{j}|T_{j}|<0.27\cdot m+O(1)$。然而，批次 $\mathcal{B}_{i}$ 中的前 $0.27\cdot m$ 次插入操作最多只能将数组 $A_{i + 2}$ 填充到 $0.54 + o(1)<0.75$ 的比例（特别要注意的是，我们不妨假设 $m = \omega(1)$）。这意味着在时间窗口 $T_{1},T_{2},\ldots$ 内的插入操作都不处于情况 3（昂贵情况）。另一方面，在时间窗口 $T_{1},T_{2},\ldots$ 完成后，该批次中剩余的插入操作都处于情况 2。因此，概率至少为 $1 - O(1/m^{2})$ 时，该批次中的插入操作都不处于情况 3。

接下来，我们对给定批次内的一次插入操作的期望搜索探测复杂度进行界定。

#### 引理 3

批次 $\mathcal{B}_{i}$ 中一次插入操作的期望搜索探测复杂度为 $O(1 + i)$。

#### 证明

批次 0 中的插入操作的搜索探测复杂度形式为 $\phi(1,j)=O(j^{2})$，其中 $j$ 是一个均值为 $O(1)$ 的几何随机变量。因此，它们的期望探测复杂度为 $O(1)$。在证明的其余部分，我们假设 $i>0$。

设 $x$ 是正在插入的元素。设 $C_{j}$（$j\in\{1,2,3\}$）是表示 $x$ 的插入操作处于情况 $j$ 的指示随机变量。设 $D_{k}$（$k\in\{1,2\}$）是表示 $x$ 最终放入数组 $A_{i + k - 1}$ 的指示随机变量。最后，设 $Q$ 是 $x$ 的搜索探测复杂度。我们可以将 $\mathbb{E}[Q]$ 分解为：

$$\begin{align*}\mathbb{E}[Q]&=\mathbb{E}[QC_{1}D_{1}]+\mathbb{E}[QC_{1}D_{2}]+\mathbb{E}[QC_{2}]+\mathbb{E}[QC_{3}]\\&\leq\mathbb{E}[QC_{1}D_{1}]+\mathbb{E}[QD_{2}]+\mathbb{E}[QC_{3}]\end{align*}$$

其中最后一个不等式使用了 $C_{2}$ 意味着 $D_{2}$ 这一事实，因此 $\mathbb{E}[QC_{1}D_{2}]+\mathbb{E}[QC_{2}]\leq\mathbb{E}[QD_{2}]$ 。

为了界定 $\mathbb{E}[QC_{1}D_{1}]$，注意到：

$$\begin{align*}\mathbb{E}[QC_{1}D_{1}]&\leq\mathbb{E}[QD_{1}\mid C_{1}]\\&=\mathbb{E}[Q\mid D_{1},C_{1}]\cdot\Pr[D_{1}\mid C_{1}]\end{align*}$$

假设在插入 $x$ 时，数组 $A_{i}$ 的填充率为 $1 - \epsilon$，并且 $x$ 使用情况 1。那么 $x$ 在 $A_{i}$ 中考虑的唯一位置是 $h_{i,1},\ldots,h_{i,f(\epsilon^{-1})}$。这些位置中有任何一个空闲的概率最多为 $O(f(\epsilon^{-1})\cdot\epsilon)$。并且，如果有一个空闲位置，那么得到的搜索探测复杂度 $Q$ 最多为 $\phi(i,f(\epsilon^{-1})\leq O(if(\epsilon^{-1})^{2})$。因此，式 (7)满足：

$$\begin{align*}\mathbb{E}[Q\mid D_{1},C_{1}]\cdot\Pr[D_{1}\mid C_{1}]&\leq O(if(\epsilon^{-1})^{2})\cdot O(f(\epsilon^{-1})\epsilon)\\&\leq O(i\epsilon f(\epsilon^{-1})^{3})\\&\leq O(i\epsilon\log^{6}\epsilon^{-1})\\&\leq O(i)\end{align*}$$

为了界定 $\mathbb{E}[QD_{2}]$，回想一下，$D_{2}$ 只有在 $A_{i + 1}$ 的填充率至多为 $0.75$ 时才会发生。因此，如果 $D_{2}$ 发生，那么 $x$ 的搜索探测复杂度为 $\phi(i + 1,j)$，其中 $j$ 至多是一个均值为 $O(1)$ 的几何随机变量。因此，我们可以将 $\mathbb{E}[QD_{2}]$ 界定为：

$$\mathbb{E}[QD_{2}]\leq\mathbb{E}[Q\mid D_{2}]\leq\mathbb{E}[\phi(i + 1,j)]$$

其中 $j$ 是一个均值为 $O(1)$ 的几何随机变量。这反过来至多为：

$$O(\mathbb{E}[i\cdot j^{2}]) = O(i)$$

最后，为了界定 $\mathbb{E}[QC_{3}]$，注意到：

$$\mathbb{E}[QC_{3}]=\mathbb{E}[Q\mid C_{3}]\cdot\Pr[C_{3}]$$

根据引理 2，我们有 $\Pr[C_{3}]=O(1/|A_{i}|^{2})$。由于情况 3 将 $x$ 插入到 $A_{i}$ 中使用的探测序列为 $h_{\phi(i,1)},h_{\phi(i,2)},\ldots$，$x$ 的搜索探测复杂度最终将由 $\phi(i,j)=O(i\cdot j^{2})$ 给出，其中 $j$ 是一个均值为 $O(|A_{i}|)$ 的几何随机变量。这意味着 $\mathbb{E}[Q\mid C_{3}]$ 的界限为 $O(i\cdot|A_{i}|^{2})$ 。因此，我们可以将式 (8)界定为：

$$\mathbb{E}[Q\mid C_{3}]\cdot\Pr[C_{3}]\leq O(i\cdot|A_{i}|^{2}/|A_{i}|^{2}) = O(i)$$

由于我们已经将式 (5) 中的每一项都界定为 $O(i)$，所以可以得出 $\mathbb{E}[Q]=O(i)$，证毕。

最后，我们将最坏情况期望插入时间界定为 $O(\log\delta^{-1})$ 。

#### 引理 4

一次插入操作的最坏情况期望时间为 $O(\log\delta^{-1})$ 。

#### 证明

批次 0 中的插入操作的期望时间为 $O(1)$，因为它们对 $A_{1}$ 进行 $O(1)$ 次期望探测。现在考虑某个批次 $\mathcal{B}_{i}$（$i\geq1$）中的插入操作。如果插入操作处于情况1或情况2，那么插入操作在 $A_{i}$ 中最多进行 $f(\delta^{-1})$ 次探测，在 $A_{i + 1}$ 中最多进行 $O(1)$ 次期望探测。因此，在这些情况下，每次插入操作的期望时间最多为 $f(\delta^{-1}) = O(\log\delta^{-1})$ 。最后，根据引理 2，每次插入操作处于情况3的概率最多为 $1/|A_{i}|^{2}$ ，并且情况 3 中的期望插入时间可以被界定为 $O(|A_{i}|)$（因为我们在 $A_{i}$ 反复探测以找到一个空闲槽位）。因此，情况 3 对期望插入时间的贡献最多为 $O(1/|A_{i}|)=O(1)$。

把这些部分整合起来，我们证明定理 1。

#### 证明

根据引理 3，$\mathcal{B}_{i}$ 中的插入操作每个都具有 $O(i)$ 的期望搜索探测复杂度。由于有 $O(\log\delta^{-1})$ 个批次，这意味着最坏情况期望搜索探测复杂度为 $O(\log\delta^{-1})$。并且，由于 $|\mathcal{B}_{i}|$ 呈几何级数下降，总体的均摊期望搜索探测复杂度为 $O(1)$。最后，根据引理 4，每次插入操作的最坏情况期望时间为 $O(\log\delta^{-1})$ 。这就完成了定理的证明。

## 3. 漏斗哈希 (Funnel Hashing)

在本节中，我们构造一种贪心开放地址法方案，该方案实现了 $O(\log^{2}\delta)$ 的最坏情况期望探测复杂度，以及 $O(\log^{2}\delta+\log\log n)$ 的高概率最坏情况探测复杂度。正如我们将看到的，高概率最坏情况界限是最优的。

#### 证明

在本节中，我们不失一般性地假设 $\delta\leq1/8$ 。设 $\alpha=\left\lceil 4\log\delta^{-1}+10\right\rceil$ 且 $\beta=\left\lceil 2\log\delta^{-1}\right\rceil$ 。

我们在本节中使用的贪心开放地址策略如下。首先，我们将数组 $A$ 拆分为两个数组，$A^{\prime}$ 和一个特殊数组 $A_{\alpha + 1}$，其中 $\left\lfloor 3\delta n/4\right\rfloor\geq|A_{\alpha + 1}|\geq\left\lceil\delta n/2\right\rceil$，具体大小的选择要使得 $|A^{\prime}|$ 能被 $\beta$ 整除。然后，将 $A^{\prime}$ 拆分为 $\alpha$ 个数组 $A_{1},\ldots,A_{\alpha}$，使得 $|A_{i}|=\beta a_{i}$，满足 $a_{i + 1}=3a_{i}/4\pm1$。也就是说，每个数组的大小是 $\beta$ 的倍数，并且它们的大小（大致）呈几何级数下降。注意，对于 $i\in[\alpha - 10]$ ，有：

$$\sum_{j>i}|A_{j}|\geq((3/4)+(3/4)^{2}+\cdots+(3/4)^{10})\cdot|A_{i}|>2.5|A_{i}|$$

每个 $i\in[\alpha]$ 的数组 $A_{i}$ 进一步细分为大小为 $\beta$ 的数组 $A_{i,j}$ 。我们定义将键 $k$ 插入到 $A_{i}$（对于 $i\in[\alpha]$ ）的尝试插入操作如下：

1. 对 $k$ 进行哈希，得到一个子数组索引 $j\in\left[\frac{|A_{i}|}{\beta}\right]$。
2. 检查 $A_{i,j}$ 中的每个槽位，看是否有空闲的。
3. 如果有空闲槽位，将其插入到第一个看到的空闲槽位中，并返回成功。否则，返回失败。

为了将键 $k$ 插入到整个数据结构中，我们依次对 $A_{1},A_{2},\ldots,A_{\alpha}$ 进行尝试插入操作，一旦成功插入就停止。这些 $\alpha = O(\log\delta^{-1})$ 次尝试中的每一次最多探测 $\beta = O(\log\delta^{-1})$ 个槽位。如果所有尝试都不成功，那么我们将 $k$ 插入到特殊数组 $A_{\alpha + 1}$ 中。特殊数组 $A_{\alpha + 1}$ 将遵循与上述不同的过程——假设其负载因子至多为 $1/4$，它将确保 $O(1)$ 的期望探测复杂度和 $O(\log\log n)$ 的最坏情况探测复杂度。在介绍 $A_{\alpha + 1}$ 的实现之前，我们先分析 $A_{1},A_{2},\ldots,A_{\alpha}$ 的行为。

从高层次上讲，我们想要证明在插入过程中每个 $A_{i}$ 都会被填充到几乎满的状态。关键是，$A_{i}$ 不需要对任何特定插入操作成功的概率给出保证。我们只希望在插入操作完成后，$A_{i}$ 中的空闲槽位少于某个值，比如说，$\delta|A_{i}|/64$ 。

#### 引理 5

对于给定的 $i\in[\alpha]$，在对 $A_{i}$ 进行 $2|A_{i}|$ 次插入尝试后，$A_{i}$ 中剩余未填充的槽位少于 $\delta|A_{i}|/64$ 的概率为 $1 - n^{-\omega(1)}$ 。

#### 证明

由于每次插入尝试从 $|A_{i}|/\beta$ 个选项中均匀随机地选择一个 $A_{i,j}$ 来使用，所以给定的 $A_{i,j}$ 被使用的期望次数是 $2\beta$。设插入到 $A_{i,j}$ 的尝试次数为 $X_{i,j}$，根据切尔诺夫界，有：

$$\Pr[X_{i,j}<\beta]=\Pr[X_{i,j}<(1 - 1/2)\mathbb{E}[X_{i,j}]]=e^{-2^{2}\beta/2}\leq e^{-4\log\delta^{-1}}=\delta^{4}\leq\frac{1}{128}\delta$$

注意，由于我们总是在选择的子数组有空闲槽位时进行插入，所以 $A_{i,j}$ 保持未填充的唯一情况是 $X_{i,j}<\beta$ 。因此，未填充子数组的期望数量最多为 $\frac{1}{128}\delta\left(\frac{|A_{i}|}{\beta}\right)$，相应地，已满子数组的期望数量为 $\left(1-\frac{1}{128}\delta\right)\left(\frac{|A_{i}|}{\beta}\right)$ 。

定义 $Y_{i,k}$ 为 $[|A_{i}|/\beta]$ 中的随机数，使得对 $A_{i}$ 的第 $k$ 次插入尝试使用子数组 $A_{i,Y_{i,k}}$ 。设 $f(Y_{i,1},\ldots,Y_{i,2|A_{i}|})$ 表示在对 $A_{i}$ 进行 $2|A_{i}|$ 次插入尝试后，$A_{i,j}$ 中仍未填充的子数组数量。改变单个 $Y_{i,k}$ 的结果最多使 $f$ 改变 $2$ ——一个子数组可能变为未填充，一个可能变为已填充。并且，根据上述内容，$\mathbb{E}[f(Y_{i,1},\ldots,Y_{i,2|A_{i}|})]=\frac{1}{128}\delta\left(\frac{|A_{i}|}{\beta}\right)$。因此，根据麦克迪尔米德不等式，有：

$$\displaystyle\Pr\left[f(Y_{i,1},\ldots,Y_{i,2|A_{i}|})\geq\frac{1}{64}\delta\left(\frac{|A_{i}|}{\beta}\right)\right]\leq\exp\left(-\frac{2\left(\frac{1}{128}\delta\frac{|A_{i}|}{\beta}\right)^{2}}{2|A_{i}|}\right)=\exp\left(-|A_{i}|O(\beta^{2}\delta^{2})\right)$$

由于 $|A_{i}|=n*{poly}(\delta)$ 且 $\delta=n^{o(1)}$，我们有：

$$|A_{i}|O(\beta^{2}\delta^{2})=n^{1 - o(1)}$$

所以 $A_{i}$ 中超过 $\frac{\delta}{64}$ 比例的子数组仍未填充的概率为 $\exp(-n^{1 - o(1)}) = 1/n^{-\omega(1)}$。所有子数组大小相同，所以，即使这些未填充的子数组完全为空，我们仍然有 $\frac{1}{64}\delta$ 比例的总槽位未填充，证毕。

作为一个推论，我们可以得到关于 $A_{\alpha + 1}$ 的以下陈述：

#### 引理 6

插入到 $A_{\alpha + 1}$ 中的键的数量少于 $\frac{\delta}{8}n$ 的概率为 $1 - n^{-\omega(1)}$ 。

#### 证明

如果对 $A_{i}$ 进行了至少 $2|A_{i}|$ 次插入尝试，则称 $A_{i}$ 被完全探索。根据引理 5，概率为 $1 - n^{-\omega(1)}$ 时，每个被完全探索的 $A_{i}$ 至少是 $(1-\delta/64)$ 满的。在本引理的其余部分，我们将基于这个性质进行条件设定。

设 $\lambda\in[\alpha]$ 是最大的索引，使得 $A_{\lambda}$ 接收到的插入尝试次数少于 $2|A_{\lambda}|$（如果不存在这样的索引，则 $\lambda=\texttt{null}$ ）。我们将处理 $\lambda$ 的三种情况。

首先，假设 $\lambda\leq\alpha - 10$ 。根据定义，我们知道对于所有 $\lambda<i\in[\alpha]$ ，$A_{i}$ 被完全探索，因此 $A_{i}$ 包含至少 $|A_{i}|(1-\delta/64)$ 个键。因此，$i>\lambda$ 的 $A_{i}$ 中的键的总数至少为：

$$(1-\delta/64)\sum_{i=\lambda + 1}^{\alpha}|A_{i}|\geq2.5(1-\delta/64)|A_{\lambda}|$$

这与对所有 $i\geq\lambda$ 的数组 $A_{i}$ 总共最多进行 $2|A_{\lambda}|$ 次插入尝试相矛盾（回想一下，根据我们算法的构造，在插入到任何 $i\geq\lambda$ 的 $A_{i}$ 之前，我们必须先尝试（并失败）插入到 $A_{\lambda}$ ）。所以这种情况是不可能的，我们完成了证明。

接下来，假设 $\alpha - 10<\lambda\leq\alpha$ 。在这种情况下，对任何 $i\geq\lambda$（包括 $i=\alpha + 1$ ）的 $A_{i}$ 尝试插入的键的数量少于 $2|A_{\alpha - 10}|<n\delta/8$，我们完成了证明。

最后，假设 $\lambda=\texttt{null}$ 。在这种情况下，每个 $i\in[\alpha]$ 的 $A_{i}$ 最多有 $\delta|A_{i}|/64$ 个空闲槽位。因此，所有插入操作结束时的空闲槽位总数最多为：

$$|A_{\alpha + 1}|+\sum_{i = 1}^{\alpha}\frac{\delta|A_{i}|}{64}=|A_{\alpha + 1}|+\frac{\delta|A^{\prime}|}{64}\leq\frac{3n\delta}{4}+\frac{n\delta}{64}<n\delta$$

这与在进行 $n(1-\delta)$ 次插入后至少有 $n\delta$ 个槽位空闲的事实相矛盾。这就完成了证明。

现在，剩下的唯一部分是实现插入到 $A_{\alpha + 1}$ 中的最多 $\leq\delta n/8$ 次插入操作。我们必须以 $O(1)$ 的期望探测复杂度和 $O(\log\log n)$ 的最坏情况探测复杂度来实现这些插入操作，同时使哈希表失败的概率至多为 $1/{poly}(n)$ 。

我们分两部分实现 $A_{\alpha + 1}$ 。也就是说，将 $A_{\alpha + 1}$ 拆分为两个大小相等（$\pm1$ ）的子数组 $B$ 和 $C$。为了插入，我们首先尝试插入到 $B$ 中，如果失败，我们插入到 $C$ 中（插入到 $C$ 中大概率会成功）。$B$ 被实现为一个均匀探测表，并且在进行 $\log\log n$ 次尝试后我们放弃在 $B$ 中的搜索。$C$ 被实现为一个双选表，每个桶的大小为 $2\log\log n$ 。

由于 $B$ 的大小 $|A_{\alpha + 1}|/2\geq\delta n/4$，其负载因子从未超过 $1/2$ 。每次插入到 $A_{\alpha + 1}$ 的操作在 $B$ 中进行 $\log\log n$ 次随机探测，每次探测成功的概率至少为 $1/2$ 。因此，给定插入操作在 $B$ 中进行的期望探测次数为 $O(1)$，并且给定插入操作尝试使用 $B$ 但失败（因此转移到 $C$ ）的概率至多为 $1/2^{\log\log n}\leq1/\log n$ 。

另一方面，$C$ 被实现为一个双选表，每个桶的大小为 $2\log\log n$ 。每次插入操作均匀随机地哈希到两个桶 $a$ 和 $b$，并使用一个探测序列，在该序列中它先尝试 $a$ 的第一个槽位，然后是 $b$ 第一个槽位，接着是 $a$ 的第二个槽位，$b$ 的第二个槽位，依此类推。这样做的效果是插入操作最终使用两个桶中较空的那个（如果两个桶一样空，则选择 $a$ ）。如果两个桶都满了，我们的表就失败了。然而，根据以下经典的双选结果 [5] ，大概率不会发生这种情况。

#### 定理 3

如果将 $m$ 个球通过为每个球均匀随机地选择两个桶并将球放入较空的桶中来放入 $n$ 个桶中，那么在 $n$ 很大时，任何桶的最大负载以高概率为 $m/n+\log\log n+O(1)$ 。

将这个定理应用到我们的场景中，我们可以得出，在 $|A_{\alpha + 1}|/\log\log n$ 很大时，因此在 $n$ 很大时，$C$ 中的任何桶都不会溢出的概率很高。这确保了我们对 $A_{\alpha + 1}$ 的实现的正确性。

由于每次插入到 $A_{\alpha + 1}$ 的操作使用 $C$ 的概率至多为 $1/\log n$，并且在 $C$ 中最多检查 $2\log\log n$ 个槽位，所以每次插入操作在 $C$ 中花费的期望时间至多为 $o(1)$。因此，到达 $A_{\alpha + 1}$ 的插入操作的期望时间为 $O(1)$，最坏情况时间为 $O(\log\log n)$ 。

由于我们对每个 $A_{i}$（单个桶）只尝试探测 $\beta$ 个槽位，给定插入操作的探测复杂度至多为 $\beta\alpha + f(A_{\alpha + 1}) = O(\log^{2}\delta^{-1}+f(A_{\alpha + 1}))$，其中 $f(A_{\alpha + 1})$ 是在 $A_{\alpha + 1}$ 中进行的探测次数。这意味着最坏情况期望探测复杂度为 $O(\log^{2}\delta^{-1})$，高概率最坏情况探测复杂度为 $O(\log^{2}\delta^{-1}+\log\log n)$ 。

我们现在只需要证明均摊期望探测复杂度。我们对每个子数组进行的期望探测次数至多为 $c\log\delta^{-1}$（包括$A_{\alpha + 1}$ ），并且我们先插入到 $A_{1}$，然后是 $A_{2}$，依此类推。因此，所有键的总期望探测复杂度至多为：

$$|A_{1}|\cdot c\log\delta^{-1}+|A_{2}|\cdot 2c\log\delta^{-1}+|A_{3}|\cdot 3c\log\delta^{-1}+\cdots+|A_{\alpha + 1}|\cdot(\alpha + 1)\log\delta^{-1}$$

由于除了 $A_{\alpha + 1}$ 之外，$A_{i}$ 的大小呈几何级数下降，而 $A_{\alpha + 1}$ 本身的大小仅为 $O(n\delta)$，上述和在（常数因子范围内）由其第一项主导。因此，所有键的总期望探测复杂度为 $O(|A_{1}|\log\delta^{-1}) = O(n\log\delta^{-1})$，这意味着均摊期望探测复杂度为 $O(n\log\delta^{-1}/(n(1-\delta)))=O(\log\delta^{-1})$，如我们所愿。这就完成了定理2的证明。

## 4. 贪心算法的下边界

在本节中，我们证明定理 2 中 $O(\log^{2}\delta^{-1})$ 的期望成本界限对于所有贪心开放地址哈希表都是最优的。

#### 定理 4

设 $n\in\mathbb{N}$ 和 $\delta\in(0,1)$ 为参数，其中 $\delta$ 是 2 的负幂次方。考虑一个容量为 $n$ 的贪心开放地址哈希表。如果将 $(1-\delta)n$ 个元素插入到该哈希表中，那么最后一次插入操作的期望时间必须为 $\Omega(\log^{2}\delta^{-1})$ 。

我们对定理4的证明将利用姚（Yao）关于均摊插入时间的下界 [21] ：

#### 命题7（姚定理 [21]）

设 $n\in\mathbb{N}$ 和 $\delta\in(0,1)$ 为参数。考虑一个容量为 $n$ 的贪心开放地址哈希表。如果将 $(1-\delta)n$ 个元素插入到该哈希表中，那么每次插入操作的均摊期望时间必须为 $\Omega(\log\delta^{-1})$ 。

基于命题7，我们可以得到以下关键引理：

#### 引理 8

存在一个通用的正常数 $c > 0$，使得以下结论成立：对于任意的 $\delta$ 和 $n$，存在某个整数 $1\leq i\leq\log\delta^{-1}$，使得第 $(1 - 1/2^{i})n$ 次插入的期望成本至少为 $ci\log\delta^{-1}$。

### 证明

设 $c$ 为一个足够小的正常数，假设引理不成立。令 $q_{j}$ 表示第 $j$ 次插入的期望成本。由于哈希表采用贪心开放地址法，我们知道 $q_{j}$ 是单调递增的。由此可得：

$$\mathbb{E}[\sum q_{j}]\geq\sum_{i\in[1,\log\delta^{-1}]}\frac{n}{2^{i}}\cdot q_{(1 - 1/2^{i})n}$$

根据假设，上式至多为：

$$\sum_{i\in[1,\log\delta^{-1}]}\frac{n}{2^{i}}\cdot c\cdot i\log\delta^{-1}\leq cn\cdot O(\log\delta^{-1})$$

将 $c$ 设置为一个足够小的正常数，这与命题 7 矛盾。

### 引理 9

设 $c$ 为引理 8 中的正常数。那么，最后一次插入的期望时间至少为：

$$\sum_{j = 1}^{\log\delta^{-1}}cj$$

### 证明

根据引理 8，存在一个整数 $1\leq i\leq\log\delta^{-1}$，使得第 $(1 - 1/2^{i})n$ 次插入的期望成本至少为 $ci\log\delta^{-1}$ 。如果 $i = \log\delta^{-1}$，那么我们就完成了证明。否则，我们可以通过对 $n$ 进行强归纳来完成证明，如下所述。

令 $S$ 表示第 $(1 - 1/2^{i})n$ 次插入后已占用位置的集合。基于 $S$ 的某个结果进行条件设定，并将未来插入的第二层成本定义为插入操作对 $[n]\setminus S$ 中的槽位进行探测的期望次数。为了分析第二层成本，我们可以想象 $[n]\setminus S$ 中的槽位是哈希表中仅有的槽位，并且 $S$ 中的槽位从每个元素的探测序列中移除。这个新的 “压缩” 哈希表大小为 $n/2^{i}$，并将接收 $n/2^{i}-\delta n=(n/2^{i})\cdot(1 - \delta 2^{i})$ 次插入操作，每次插入都采用贪心开放地址法。通过归纳法可知，“压缩” 哈希表中的最后一次插入的期望成本至少为：

$$\sum_{j = 1}^{\log\delta^{-1}-i}cj$$  (9)

这相当于说完整哈希表中的最后一次插入的期望第二层成本至少为 (9)。此外，尽管 (9)是基于 $S$ 的某个特定结果建立的，但由于它对每个单独的结果都成立，所以在没有任何条件设定的情况下也成立。

最后，除了第二层成本，最后一次插入甚至在找到任何不在 $S$ 中的槽位时，都必须至少进行 $ci\log n$ 次期望探测（这是由于我们之前应用了引理 8）。因此，最后一次插入所产生的总期望成本至少为：

$$ci\log\delta^{-1}+\sum_{j = 1}^{\log\delta^{-1}-i}cj\geq\sum_{j = 1}^{\log\delta^{-1}}cj$$

正如我们所期望的。

最后，我们可以作为引理 9 的推论来证明定理 4。

### 定理 4 的证明

根据引理 9，存在一个正常数 $c$，使得最后一次插入所产生的期望成本至少为：

$$\sum_{j = 1}^{\log\delta^{-1}}cj=\Omega(\log^{2}\delta^{-1})$$

## 5. 无重排开放地址哈希表的下边界

在本节中，我们给出两个下界，它们不仅适用于贪心开放地址哈希表，而且适用于任何不执行重排操作的开放地址哈希表。我们的第一个结果是对最坏情况期望探测复杂度给出 $\Omega(\log\delta^{-1})$ 的下界（与定理 1 中的上界相匹配）。我们的第二个结果是对（高概率）最坏情况探测复杂度给出 $\Omega(\log^{2}\delta^{-1}+\log\log n)$ 的下界（与定理 2 中的上界相匹配，定理 2 中的上界是由一种贪心方案实现的）。

在以下证明中，我们假设键的探测序列是独立同分布的随机变量。这等同于假设全域大小是一个大的多项式，然后随机（有放回）采样键；在高概率下，这样的采样过程不会对任何键进行两次采样。

### 5.1 通用定义

这两个下界都将使用一组共享的定义：

- 令 $m = n(1 - \delta)$。
- 令 $k_{1},k_{2},\ldots,k_{m}$ 为要插入的键的集合。
- 令 $H_{i}(k_{j})$ 为键 $k_{j}$ 的探测序列中的第 $i$ 个元素。由于 $H_{i}(k_{j})$ 的分布对于所有 $k_{j}$ 都是相同的，我们有时会使用 $H_{i}$ 作为简写。我们也会使用 $h_{i}$ 来表示 $H_{i}$ 的一个（非随机）特定结果。
- 令 $\mathcal{H}_{c}(k_{j})=\{H_{i}(k_{j}):i\in[c]\}$ 表示键 $k_{j}$ 进行的前 $c$ 次探测所组成的集合。同样，由于 $\mathcal{H}_{c}(k_{j})$ 对于所有 $k_{j}$ 具有相同的分布，我们有时会使用 $\mathcal{H}_{c}$ 作为简写。
- 对于 $i\in[m]$，令 $S_{i}\subset[n]$，$|S_{i}| = n - i$ 为一个随机变量，表示插入 $i$ 个键后数组中未填充的槽位集合（其分布由哈希方案导出）。
- 对于 $i\in[m]$ 和 $j\in\mathbb{N}$，令 $X_{i,j}$ 为一个随机变量，用于指示在插入 $k_{i}$ 时，由 $H_{j}(k_{i})$ 索引的槽位是否为空。
- 令 $Y_{i}$ 为键 $k_{i}$ 在探测序列中使用的位置。换句话说，键 $k_{i}$ 被放置在槽位 $H_{Y_{i}}(k_{i})$ 中。我们也会使用 $y_{i}$ 来表示 $Y_{i}$ 的一个（非随机）特定结果。注意，该槽位必须为空，所以 $Y_{i}\in\{r:X_{i,r}=1\}$（在贪心算法中，会选择第一个可用的槽位 —— 在这种情况下，$Y_{i}=\min\{r:X_{i,r}=1\}$ ，但我们对算法是否贪心不做任何假设）。
- 令 $L_{i}$ 为随机变量，表示第 $i$ 个键插入到数组中的位置。

### 5.2 最坏情况期望探测复杂度

在本节中，我们证明以下定理：

#### 定理 5

在任何不重排且达到负载因子 $(1 - \delta)$ 的开放地址方案中，最坏情况期望探测复杂度必须为 $\Omega(\log\delta^{-1})$ 。特别地，存在某个 $i\in[m]$ ，使得 $\mathbb{E}[Y_{i}]=\Omega(\log\delta^{-1})$ 。

从高层次上讲，我们想要反转上界算法的思路。与将数组划分为大小呈指数递减的子数组不同，我们证明至少在某种程度上，这样的构造是自然出现的。给定最坏情况期望探测复杂度的上界 $c$，我们将证明必然存在不相交的槽位组 $v_{1},v_{2},\ldots,v_{\Theta(\log\delta^{-1})}$，其大小呈指数递减，并且对于每个 $i$，有 $\mathbb{E}[\mathcal{H}_{2c}\cap v_{i}]\geq\Omega(1)$ 。这反过来意味着 $2c\geq\mathbb{E}[|\mathcal{H}_{2c}|]\geq\Omega(\log\delta^{-1})$ 。正如我们将看到的，棘手的部分是以一种保证该性质的方式定义 $v_{i}$ 。

### 证明

令 $c$ 为对所有 $i$ 都成立的 $E[Y_{i}]$ 的任意上界。我们想要证明 $c=\Omega(\log\delta^{-1})$ 。注意，根据马尔可夫不等式，对于任何 $i\in[m]$，有 $\Pr[Y_{i}\leq 2c]\geq\frac{1}{2}$ 。

令 $\alpha=\left\lfloor\frac{\log\delta^{-1}}{3}\right\rfloor\in\Omega(\log\delta^{-1})$ 。对于 $i\in[\alpha]$ ，令 $a_{i}=n\left(1-\frac{1}{2^{3i}}\right)$ 。注意 $a_{i}\leq a_{\log\delta^{-1}/3}\leq n\left(1-\frac{1}{2^{3\log\delta^{-1}/3}}\right)=n(1 - \delta)=m$ 。进一步注意 $|S_{a_{i}}| = n - a_{i}=\frac{n}{2^{3i}}$，因为 $S_{i}$ 表示仍未填充的槽位；并且注意 $|S_{a_{i}}|$ 的大小，对于 $i = 1,2,\ldots$ ，形成一个公比为 $1/2^{3}=1/8$ 的几何序列。由此可知，对于任何 $s_{a_{i}}\leftarrow S_{a_{i}}$，$s_{a_{i + 1}}\leftarrow S_{a_{i + 1}},\ldots$ ，$s_{a_{\alpha}}\leftarrow S_{a_{\alpha}}$ ，即使 $s_{a_{i}}$ 之间不兼容（即 $s_{a_{i + 1}}\not\subseteq s_{a_{i}}$ ），我们有：

$$\left|s_{a_{i + 1}}\cup s_{a_{i + 2}}\cup\cdots\cup s_{a_{\alpha}}\right|\leq\sum_{j\geq i + 1}|s_{a_{j}}|\leq|s_{a_{i}}|/7$$

由于 $\Pr[Y_{i}\leq 2c]\geq\frac{1}{2}$ ，对于任何 $t < 2m - n = n(1 - 2\delta)$，我们有：

$$\mathbb{E}[|\{i:Y_{i}\leq 2c\text{ 且 }t < i\leq m\}|]\geq\frac{m - t}{2}\geq\frac{n - t}{4}=\frac{|S_{t}|}{4}$$

因此，对于每个 $j\in[\alpha - 1]$ ，存在某个 $s_{a_{j}}\subseteq[n]$，使得：

$$\displaystyle\mathbb{E}\left[|\{i:Y_{i}\leq 2c\text{ 且 }a_{j}<i\leq m\}|\,\bigg{|}\,S_{a_{j}}=s_{a_{j}}\right]\geq\frac{|s_{a_{j}}|}{4}$$  (10)

也就是说，我们找到了随机变量 $S_{a_{j}}$ 的某个具体实例 $s_{a_{j}}$，使得在 $S_{a_{j}} = s_{a_{j}}$ 的条件下，$i > a_{j}$ 时 $Y_{i}$ 的 “小” 值的期望数量至少为总体期望数量。需要注意的是，$s_{a_{1}},s_{a_{2}},\ldots$ 之间可能存在任意关系；它们作为 $S_{a_{1}},S_{a_{2}},\ldots$ 的值不必相互兼容。也许令人惊讶的是，即便如此，我们仍然能够推断出 $s_{a_{i}}$ 之间的关系。特别是，我们将在证明结束时表明，对于每个 $j$ 和每次插入，在前 $2c$ 次探测中探测到 $s_{a_{j}}\setminus\bigcup_{k > j}s_{a_{k}}$ 中位置的期望次数为 $\Omega(1)$。这将使我们推断出，在期望情况下，探测序列的前 $2c$ 个元素至少包含 $\Omega(\log\delta^{-1})$ 个不同的值，从而意味着 $c=\Omega(\log\delta^{-1})$ 。

令：

$$\displaystyle\mathcal{L}_{j}=\{L_{i}:m\geq i > a_{j}\text{ 且 }Y_{i}\leq 2c\}\,\bigg{|}\,S_{a_{j}}=s_{a_{j}}$$

为（随机变量）在 $i > a_{j}$ 且 $S_{a_{j}} = s_{a_{j}}$ 的条件下，“快速” 插入（即满足 $Y_{i}\leq 2c$ 的插入）所使用的位置集合。注意，根据(10)，有 $\mathbb{E}[|\mathcal{L}_{j}|]\geq\frac{|s_{a_{j}}|}{4}$ 。观察到 $\mathcal{L}_{i}\subseteq s_{a_{j}}$，因为从 $s_{a_{j}}$ 作为空槽位集合开始填充的所有槽位都必须来自 $s_{a_{j}}$ 。我们现在将论证，由于 $\mathbb{E}[|\mathcal{L}_{j}|]$ 如此之大，我们可以保证 $\mathbb{E}[|\mathcal{L}_{j}\setminus\bigcup_{k > j}s_{a_{k}}|]$ 也很大，即 $\Omega(|s_{a_{j}}|)$ 。

定义：

$$\begin{array}{lcl}t_{j}&=&\bigcup_{k > j}s_{a_{k}}\\v_{j}&=&s_{a_{j}}\setminus t_{j}\end{array}$$

并注意 $v_{j}$ 是不相交的：

#### 声明 10

对于所有 $j\neq k$ ，有 $v_{j}\cap v_{k}=\emptyset$ 。也就是说，所有 $v_{j}$ 都是相互不相交的。

#### 证明

不失一般性，假设 $j < k$。根据 $t_{j}$ 的定义，$s_{a_{k}}\subseteq t_{j}$ 。根据 $v_{k}$的定义，$v_{k}\subseteq s_{a_{k}}$ 。最后，根据 $v_{j}$ 的定义，$v_{j}\cap t_{j}=\emptyset$。因此，$v_{j}\cap v_{k}=\emptyset$ 。

正如我们前面看到的，$|t_{j}| = |s_{a_{j + 1}}\cup\cdots\cup s_{a_{\alpha}}|\leq|s_{a_{j}}|/7$。由于$\mathcal{L}_{i}\subseteq s_{a_{j}}\subseteq v_{j}\cup t_{j}$，我们有：

$$\displaystyle\frac{\left|s_{a_{j}}\right|}{4}\leq\mathbb{E}\left[\left|\mathcal{L}_{i}\right|\right]=\mathbb{E}\left[\left|\mathcal{L}_{i}\cap v_{j}\right|\right]+\mathbb{E}\left[\left|\mathcal{L}_{i}\cap t_{j}\right|\right]\leq\mathbb{E}\left[\left|\mathcal{L}_{i}\cap v_{j}\right|\right]+\frac{\left|s_{a_{j}}\right|}{7}$$

相减可得：

$$\displaystyle\mathbb{E}\left[\left|\mathcal{L}_{i}\cap v_{j}\right|\right]\geq\frac{\left|s_{a_{j}}\right|}{4}-\frac{\left|s_{a_{j}}\right|}{7}\geq\frac{\left|s_{a_{j}}\right|}{16}$$  (11)

证明的其余部分的高层次思路如下。我们想要论证 $v_{j}$ 是不相交的集合，每个集合都有相当大（即 $\Omega(1)$ ）的概率在给定探测序列的前 $2c$ 次探测 $\mathcal{H}_{2c}$ 中出现一个元素。由此，我们将能够推断出$c$在渐近意义上至少与 $v_{j}$ 的数量一样大，而 $v_{j}$ 的数量为 $\Omega(\log\delta^{-1})$ 。

令：

$$\begin{array}{lcl}p_{i,j}&=&\Pr[Y_{i}\leq 2c\text{ 且 }L_{i}\in v_{j}]\\q_{j}&=&\Pr[\mathcal{H}_{2c}\cap v_{j}\neq\emptyset]\end{array}$$

我们必然有 $p_{i,j}\leq q_{j}$ ，因为对于 $Y_{i}\leq 2c$ 且 $L_{i}\in v_{j}$，我们必然有至少一个哈希函数在前 $2c$ 次输出中产生了 $vj$ 中的一个索引。因此我们有：

$$\frac{|s_{a_{j}}|}{16}\leq\mathbb{E}[|\mathcal{L}_{j}\cap v_{j}|]=\sum_{i = a_{j}+1}^{m}p_{i,j}\leq\sum_{i = a_{j}+1}^{m}q_{j}=q_{j}(m - a_{j})\leq q_{j}(n - a_{j})=q_{j}|s_{a_{j}}|$$

由此我们得出，对于所有 $j∈[α - 1]$，$qj≥1/16$ 。因此，

$$\begin{align*}2c&=|\{H_{i}:i\in[2c]\}|\\&=|\{H_{i}:i\in[2c]\}\cap[n]|\\&=\mathbb{E}[|\{H_{i}:i\in[2c]\}\cap[n]|]\\&\geq\sum_{j = 1}^{\alpha - 1}\mathbb{E}[|\{H_{i}:i\in[2c]\}\cap v_{j}|]\\&\geq\sum_{j = 1}^{\alpha - 1}q_{j}\\&\geq\frac{1}{16}(\alpha - 1)=\Omega(\log\delta^{-1})\end{align*}$$

证毕。

### 5.3 高概率最坏情况探测复杂度

为了证明高概率下边界，我们将使用与上一个证明类似的集合构造。唯一的区别是我们现在对最大探测复杂度有了一个上限。就定理 5 证明中使用的变量而言，这一额外的约束使我们能够得到关于 $\mathbb{E}[\mathcal{H}_{2c}\cap v_{j}]$ 的更强的界——具体来说，我们对这个量的界将从 $\Omega(1)$ 增加到 $\Omega(\log\delta^{-1})$ 。

主要思想是，由于我们现在对一次插入可以使用的探测次数有了一个最坏情况的上限 c，我们可以更明确地分析一个特定槽位被探测到的实际概率。正如将展示的，为了使一个槽位以大于 $(1 - δ)$ 的概率被看到（这是填充 $(1 - δ)$ 比例的槽位所必需的），它必须以至少 $\Omega(\log\delta^{-1}/n)$ 的概率出现在 $\mathcal{H}_{c}$ 中。将这一点整合到我们的分析中，与定理  的证明相比，我们将能够获得一个额外的 $\Omega(\log\delta^{-1})$ 因子。

#### 定理 6

在任何不执行重排的开放地址方案中，概率大于 1/2 时，必然存在某个键，其探测复杂度最终为 $\Omega(\log^{2}\delta^{-1})$ 。换句话说，对于所有 $c∈o$ $(\log^{2}\delta^{-1})$ ，有：

$$\Pr\left[Y_{i}\leq c\,\forall i\in[m]\right]\leq\frac{1}{2}$$

### 证明

假设存在一种不执行重排的开放地址方案，对于某个 $c∈\mathbb{N}$，所有键的探测复杂度以大于 1/2 的概率至多为 $c$ 。我们将证明 $c=\Omega(\log^{2}\delta^{-1})$ 。

根据定义，我们有：

$$\displaystyle\Pr[Y_{i}\leq c\,\forall i\in[m]]>\frac{1}{2}$$  (12)

因此，对于每个 $i∈{0, 1, …, m}$，必然存在某个 $si⊆[n]$，其大小为 $(n - i)$，使得：

$$\Pr\left[Y_{i}\leq c\,\forall i\in[m]\,\bigg{|}\,S_{i}=s_{i}\right]>\frac{1}{2}$$

否则，根据条件概率的定义，我们将与 (12) 矛盾。

#### 声明 11

对于任何 $i < n(1 - 256δ)$，必然存在某个集合 $ri⊆si⊆[n]$，其大小 $|ri| > (n - i)/2 = |si|/2$ ，使得对于任何 $x∈ri$，有：

$$P[x\in\mathcal{H}_{c}]>\frac{1}{32}\frac{\log\left(\frac{|s_{i}|}{n\delta}\right)}{|s_{i}|}$$

### 证明

一个负载因子为 $(1 - δ)$ 的哈希表成功（即所有键的探测复杂度至多为 $c$）的必要条件是：

$$s_{i}\setminus\cup_{j>i}\mathcal{H}_{c}(k_{j})$$

的大小至多为 $δn$ 。实际上，这些是在插入 $ki$ 之后为空且在剩余插入的（前 $c$ 次探测）中从未被探测到的槽位集合。

因此，

$$\displaystyle\Pr\left[\left|s_{i}\cap\left(\bigcup_{j=i + 1}^{m}\mathcal{H}_{c}(k_{j})\right)\right|>|s_{i}|-\delta n:S_{i}=s_{i}\right]>\frac{1}{2}$$  (13)

注意，对 $Si = si$ 的条件设定是不必要的，因为随机变量 $\mathcal{H}_{c}(k_{j})$，$j > i$，与事件 $Si = si$ 是独立的。

令 $p=\frac{\log\left(\frac{|s_{i}|}{n\delta}\right)}{|s_{i}|}$ ，令 $ti$ 为所有 $x∈si$ 的集合，使得：

$$\Pr[x\in\mathcal{H}_{c}]\leq\frac{p}{32}$$

我们将通过证明 $|ti| < |si|/2$ 来完成声明的证明。为此，我们将计算 $ti$ 中预期出现在某个 $H_{c}(k_{j})$（j > i）中的元素数量：

$$\begin{align*}\mathbb{E}\left[\left|t_{i}\cap\left(\bigcup_{j=i + 1}^{m}\mathcal{H}_{c}(k_{j})\right)\right|\right]&=\sum_{x\in t_{i}}\Pr\left[x\in\bigcup_{j=i + 1}^{m}\mathcal{H}_{c}(k_{j})\right]\\&=\sum_{x\in t_{i}}1-\left(1-\Pr\left[x\in\mathcal{H}_{c}\right]\right)^{m - i}\text{ (因为 }\mathcal{H}_{c}(k_{j})\text{ 对 }j\text{ 是独立同分布的)}\\&\leq\sum_{x\in t_{i}}1-\left(1-\frac{p}{32}\right)^{|s_{i}|-n\delta}\\&\leq\sum_{x\in t_{i}}1-\left(1-\frac{p}{32}\right)^{|s_{i}|/2}\text{ (因为根据假设 }i < n(1 - 2\delta)\text{)}\\&\leq|t_{i}|-|t_{i}|\left(1-\frac{1}{|s_{i}|}\right)^{\log\left(\frac{|s_{i}|}{n\delta}\right)|s_{i}|/64}\text{ (因为如果 }x,t\geq1\text{，则 }(1 - x/t)\leq(1 - 1/t)^{x}\text{)}\\&<|t_{i}|-|t_{i}|(1/2)^{\log\left(\frac{|s_{i}|}{n\delta}\right)/8}\\&<|t_{i}|-|t_{i}|\left(\frac{n\delta}{|s_{i}|}\right)^{1/8}\end{align*}$$

根据假设，$i < n(1 - 256δ)$，所以 $|si| > n - n(1 - 256δ) = 256nδ$。由于 $|si| > 256nδ$，我们有 $\left(\frac{n\delta}{|s_{i}|}\right)^{1/8}<\left(\frac{1}{256}\right)^{1/8}=\frac{1}{2}$ 。因此我们有：

$$|t_{i}|-|t_{i}|\left(\frac{n\delta}{|s_{i}|}\right)^{1/8}<\frac{|t_{i}|}{2}$$

为了使哈希表插入过程成功，我们必须有：

$$\mathbb{E}\left[\left|t_{i}\cap\left(\bigcup_{j=i + 1}^{m}\mathcal{H}_{c}(k_{j})\right)\right|\right]\geq|t_{i}|-n\delta$$

否则，超过 $nδ$ 个槽位将永远不会成为任何哈希函数输出的一部分，因此在最后必然是未填充的。由此可知 $|ti| < 2nδ < |si|/2$，证毕。

令 $ai = n(1 - 1/4i)$ ，对于 $i∈[\log\delta^{-1}/4]$ 。观察到，对于任何 $i∈[\log\delta^{-1}/4]$ ，有：

$$\left|\bigcup_{j=i + 1}^{\log\delta^{-1}/4}s_{a_{j}}\right|\leq\sum_{j = 1}^{\log\delta^{-1}/4 - i}\frac{\left|s_{a_{i}}\right|}{4^{j}}\leq\frac{|s_{a_{i}}|}{3}\leq\frac{3|s_{a_{i}}|}{8}$$

令：

$$v_{i}=r_{a_{i}}\setminus\left(\bigcup_{j=i + 1}^{\log\delta^{-1}/4}s_{a_{j}}\right)$$

对于 $i∈[\log\delta^{-1}/4]$ 。根据声明 11，我们有 $|ra_{i}|≥|sa_{i}|/2$，所以：

$$\left|v_{i}\right|\geq\frac{\left|s_{a_{i}}\right|}{2}-\frac{3\left|s_{a_{i}}\right|}{8}=\frac{\left|s_{a_{i}}\right|}{8}$$

注意，对于所有 $i≠j$，$vi∩vj = ∅$，其证明与上一小节的声明 10 类似。此外，注意到：

$$|s_{a_{i}}|\geq n\left(\frac{1}{4}\right)^{\log\delta^{-1}/4}=n\left(\frac{1}{2}\right)^{\log\delta^{-1}/2}=n\sqrt{\delta}$$

我们现在通过展开每个 $vi$ 的定义来获得对 $|\mathcal{H}_{c}|≤c$ 的下界。特别地，在不失一般性地假设 $\log\delta^{-1}/4>256$ 的情况下，我们有：

$$\begin{align*}c&\geq\mathbb{E}[|\mathcal{H}_{c}|]\\&\geq\sum_{i = 1}^{\log\delta^{-1}/4}\mathbb{E}[|\mathcal{H}_{c}\cap v_{i}|]\\&=\sum_{i = 1}^{\log\delta^{-1}/4}\sum_{x\in v_{i}}\mathbb{E}[|\{x\}\cap\mathcal{H}_{c}|]\\&=\sum_{i = 1}^{\log\delta^{-1}/4}\sum_{x\in v_{i}}\Pr[x\in\mathcal{H}_{c}]\\&\geq\sum_{i = 1}^{\log\delta^{-1}/4}\sum_{x\in v_{i}}\frac{1}{32}\frac{\log\left(\frac{|s_{a_{i}}|}{n\delta}\right)}{|s_{a_{i}}|}\\&=\frac{1}{32}\sum_{i = 1}^{\log\delta^{-1}/4}|v_{i}|\frac{\log\left(\frac{|s_{a_{i}}|}{n\delta}\right)}{|s_{a_{i}}|}\\&\geq\frac{1}{32}\sum_{i = 1}^{\log\delta^{-1}/4}\frac{|s_{a_{i}}|}{8}\frac{\log\left(\frac{n\sqrt{\delta}}{n\delta}\right)}{|s_{a_{i}}|}\\&=\frac{1}{32}\sum_{i = 1}^{\log\delta^{-1}/4}\frac{\log\delta^{-1}}{16}\\&=\frac{1}{32}\cdot\frac{1}{16}\cdot\frac{1}{4}\log^{2}\delta^{-1}=\Omega(\log^{2}\delta^{-1})\end{align*}$$

正如我们所期望的。

我们现在将上述结果与一个已知结果相结合，以获得我们的完整下边界：

#### 定理 7

在任何不支持重排的开放地址方案中，在假设 $(1 - δ)=\Omega(1)$ 的情况下，概率大于 1/2 时，必然存在某个键，其探测复杂度最终为 $\Omega(\log\log n+\log^{2}\delta^{-1})$ 。

### 证明

我们只需要证明存在某个键的探测复杂度为 $\Omega(\log\log n)$，因为我们已经在定理 6 中证明了存在某个键的探测复杂度为 $\Omega(\log^{2}\delta^{-1})$ 。我们的证明模仿了 [3] 中定理 5.2 的证明，而 [3] 中的证明主要基于 [20] 中的以下定理：

#### 定理 8（[20]中的定理2）

假设 $m$ 个球通过任意机制依次放入 $m$ 个箱子中，唯一的限制是每个球根据 $[m]d$ 上的某个任意分布在 $d$ 个箱子中进行选择。那么在这个过程结束时，最满的箱子中以高概率有 $\Omega(\log\log n/d)$ 个球。

现在，假设我们有某种任意的不执行重排的开放地址方案，其中概率大于 1/2 时，所有键的探测复杂度至多为 d。我们将我们的哈希表方案修改为一个球和箱子的过程，使得每次选择最多 d 个箱子，具体如下。

假设键 $ki$ 被插入到位置 $li = hj(ki)$ 。如果 $j≤d$，则将球 $i$ 放入箱子 $hj(ki) mod m$ 中。否则，将球 $i$ 放入箱子 $hd(ki) mod m$ 中，通过这种方式确保该方案每次选择最多 $d$ 个箱子；可能的选择集合是：$\{H_{j}(k_{i})\text{ mod }m:j\leq d\}$，其大小（最多）为 $d$ 。这个过程还确保了最满的箱子中球的数量很可能很少：

#### 引理 12

在这个过程结束时，概率大于 1/2 时，最满的箱子中有 $O(1)$ 个球。

### 证明

假设球 $i$ 落入箱子 $Bi$ 中。那么落入第 $j$ 个箱子的球的索引集为 $\{i:B_{i}=j\}$。

现在，假设对于所有 $i∈[m]$，$Bi = Li mod m$，并且注意到这种情况发生的概率大于 1/2。由于每个槽位只能存储一个键，对于任何 $i≠j$，$Li≠Lj$ 。因此，$\{i:B_{i}=j\}\subseteq\{i\in[n]:i\text{ mod }m = j\}$ 。由于 (1 - δ)=\Omega(1)，我们有 m=\Omega(n)，即 n = O(m)。因此，对于所有 i∈[m]，$|\{i\in[n]:i\text{ mod }m = j\}| = O(1)$，并且最满的箱子中最多有 $O(1)$ 个球，证毕。

根据定理 8，在这个过程结束时，最满的箱子中以高概率有 $\Omega(\log\log n/d)$ 个球。如果 $d = o(\log\log n)$，那么最满的箱子中以高概率有 $\omega(1)$ 个球，这与引理 12 矛盾。因此，$d=\Omega(\log\log n)$，正如我们所期望的。

## 6. 致谢和资金支持

作者们感谢米克尔·索鲁普（Mikkel Thorup）进行的多次有益讨论，包括对本文早期版本的反馈。

威廉·库兹莫尔（William Kuszmaul）部分得到了哈佛拉宾博士后奖学金以及美国国家科学基金会（NSF）资助的哈佛FODSI奖学金（项目编号：DMS - 2023528）的支持。马丁·法拉赫 - 科尔顿（Martín Farach - Colton）部分得到了伦纳德·J·舒斯特克教授职位以及NSF资助（项目编号：CCF - 2106999、NSF - BSF CCF - 2247576、CCF - 2423105和CCF - 2420942）的支持。

## 参考文献

[1] Miklós Ajtai, János Komlós, and Endre Szemerédi. There is no fast single hashing algorithm. Information Processing Letters, 7(6):270–273, 1978.

[2] Michael A Bender, Alex Conway, Martín Farach - Colton, William Kuszmaul, and Guido Tagliavini. Iceberg hashing: Optimizing many hash - table criteria at once. Journal of the ACM, 70(6):1–51, 2023.

[3] Michael A. Bender, Alex Conway, Martín Farach - Colton, William Kuszmaul, and Guido Tagliavini. Tiny Pointers, pages 477–508. 2023. doi:10.1137/1.9781611977554.ch21.

[4] Michael A. Bender, Martin Farach - Colton, Simai He, Bradley C. Kuszmaul, and Charles E. Leiserson. Adversarial contention resolution for simple channels. In SPAA, pages 325–332. ACM, 2005.

[5] Petra Berenbrink, Artur Czumaj, Angelika Steger, and Berthold Vöcking. Balanced allocations: the heavily loaded case. In Proceedings of the Thirty - Second Annual ACM Symposium on Theory of Computing, STOC ’00, page 745–754. Association for Computing Machinery, 2000. doi:10.1145/335305.335411.

[6] Richard P Brent. Reducing the retrieval time of scatter storage techniques. Communications of the ACM, 16(2):105–109, 1973.

[7] Andrei Z Broder and Anna R Karlin. Multilevel adaptive hashing. In Proceedings of the first annual ACM - SIAM symposium on Discrete algorithms, pages 43–53, 1990.

[8] Walter A Burkhard. External double hashing with choice. In 8th International Symposium on Parallel Architectures, Algorithms and Networks (ISPAN’05), pages 8–pp. IEEE, 2005.

[9] Dimitris Fotakis, Rasmus Pagh, Peter Sanders, and Paul Spirakis. Space efficient hash tables with worst case constant access time. Theory of Computing Systems, 38(2):229–248, 2005.

[10] Gaston H Gonnet and J Ian Munro. Efficient ordering of hash tables. SIAM Journal on Computing, 8(3):463–478, 1979.

[11] Donald E Knuth. Notes on “open” addressing. Unpublished memorandum, pages 11–97, 1963.

[12] Donald E Knuth. Computer science and its relation to mathematics. The American Mathematical Monthly, 81(4):323–343, 1974.

[13] Donald Ervin Knuth. The Art of Computer Programming, Volume III: Sorting and Searching. Addison - Wesley, 2nd edition, 1998. URL: https://www.worldcat.org/oclc/312994415.

[14] George Lueker and Mariko Molodowitch. More analysis of double hashing. In Proceedings of the twentieth annual ACM symposium on Theory of computing, pages 354–359, 1988.

[15] Paul M Martini and Walter A Burkhard. Double hashing with multiple passbits. International Journal of Foundations of Computer Science, 14(06):1165–1182, 2003.

[16] Mariko Molodowitch. Analysis and design of algorithms: double hashing and parallel graph searching. University of California, Irvine, 1990.

[17] J Ian Munro and Pedro Celis. Techniques for collision resolution in hash tables with open addressing. In Proceedings of 1986 ACM Fall joint computer conference, pages 601–610, 1986.

[18] Peter Sanders. Hashing with linear probing and referential integrity. arXiv preprint arXiv:1808.04602, 2018.

[19] Jeffrey D Ullman. A note on the efficiency of hashing functions. Journal of the ACM (JACM), 19(3):569–575, 1972.

[20] Berthold Vöcking. How asymmetry helps load balancing. Journal of the ACM (JACM), 50(4):568–589, 2003.

[21] Andrew C Yao. Uniform hashing is optimal. Journal of the ACM (JACM), 32(3):687–693, 1985.

**中文**

[1] Miklós Ajtai、János Komlós和Endre Szemerédi。没有快速的单一哈希算法。信息处理快报，7（6）：270–2731978。

[2] 迈克尔·A·本德、亚历克斯·康威、马丁·法拉赫·科尔顿、威廉·库兹马尔和吉多·塔利阿维尼。冰山哈希：同时优化多个哈希表标准。ACM杂志，70（6）：1-512023。

[3] 迈克尔·A·本德、亚历克斯·康威、马丁·法拉赫·科尔顿、威廉·库兹马尔和吉多·达格利阿维尼。《小指针》，第477-508页。2023年。电话：10.1137/1.9781611977554.ch21。

[4] 迈克尔·A·本德、马丁·法拉赫·科尔顿、何思迈、布拉德利·C·库兹马尔和查尔斯·E·莱瑟森。简单信道的对抗性争用解决方案。SPAA，第325-332页。ACM，2005年。

[5] 佩特拉·贝伦布林克（Petra Berenbrink）、阿图尔·楚马伊（Artur Czumaj）、安杰利卡·斯特格（Angelika Steger）和贝特霍尔德·弗金（Berthold Vöcking）。《平衡分配：高负载情况》。收录于《第三十二届ACM计算理论年会论文集》（Proceedings of the Thirty - Second Annual ACM Symposium on Theory of Computing），2000年，第745 - 754页。美国计算机协会（Association for Computing Machinery）。doi:10.1145/335305.335411。

[6] 理查德·P·布伦特（Richard P Brent）。《降低散列存储技术的检索时间》。《ACM通讯》（Communications of the ACM），1973年，第16卷，第2期，第105 - 109页。

[7] 安德烈·Z·布罗德（Andrei Z Broder）和安娜·R·卡林（Anna R Karlin）。《多级自适应哈希》。收录于《第一届ACM - SIAM离散算法年会论文集》（Proceedings of the first annual ACM - SIAM symposium on Discrete algorithms），1990年，第43 - 53页。

[8] 沃尔特·A·伯克哈德（Walter A Burkhard）。《带选择的外部双重哈希》。收录于《第八届并行架构、算法和网络国际研讨会论文集》（8th International Symposium on Parallel Architectures, Algorithms and Networks (ISPAN’05)），2005年，第8页起。IEEE。

[9] 迪米特里斯·福塔基斯（Dimitris Fotakis）、拉斯穆斯·帕格（Rasmus Pagh）、彼得·桑德斯（Peter Sanders）和保罗·斯皮拉基斯（Paul Spirakis）。《具有最坏情况常数访问时间的高效空间哈希表》。《计算系统理论》（Theory of Computing Systems），2005年，第38卷，第2期，第229 - 248页。
[10] 加斯顿·H·贡内（Gaston H Gonnet）和J·伊恩·蒙罗（J Ian Munro）。《哈希表的高效排序》。《SIAM计算杂志》（SIAM Journal on Computing），1979年，第8卷，第3期，第463 - 478页。

[11] 唐纳德·E·克努特（Donald E Knuth）。《关于“开放”寻址的笔记》。未发表的备忘录，1963年，第11 - 97页。

[12] 唐纳德·E·克努特（Donald E Knuth）。《计算机科学及其与数学的关系》。《美国数学月刊》（The American Mathematical Monthly），1974年，第81卷，第4期，第323 - 343页。

[13] 唐纳德·欧文·克努特（Donald Ervin Knuth）。《计算机程序设计艺术 第三卷：排序与搜索》（The Art of Computer Programming, Volume III: Sorting and Searching）。艾迪生 - 韦斯利（Addison - Wesley），第二版，1998年。网址：https://www.worldcat.org/oclc/312994415。

[14] 乔治·勒克（George Lueker）和真理子·莫洛多维奇（Mariko Molodowitch）。《双重哈希的更多分析》。收录于《第二十届ACM计算理论年会论文集》（Proceedings of the twentieth annual ACM symposium on Theory of computing），1988年，第354 - 359页。

[15] 保罗·M·马蒂尼（Paul M Martini）和沃尔特·A·伯克哈德（Walter A Burkhard）。《带多个传递位的双重哈希》。《国际计算机科学基础杂志》（International Journal of Foundations of Computer Science），2003年，第14卷，第06期，第1165 - 1182页。

[16] 真理子·莫洛多维奇（Mariko Molodowitch）。《算法分析与设计：双重哈希和并行图搜索》。加利福尼亚大学欧文分校，1990年。

[17] J·伊恩·蒙罗（J Ian Munro）和佩德罗·塞利斯（Pedro Celis）。《开放寻址哈希表中的冲突解决技术》。收录于《1986年ACM秋季联合计算机会议论文集》（Proceedings of 1986 ACM Fall joint computer conference），1986年，第601 - 610页。

[18] 彼得·桑德斯（Peter Sanders）。《带线性探测和引用完整性的哈希》。arXiv预印本arXiv:1808.04602，2018年。

[19] 杰弗里·D·厄尔曼（Jeffrey D Ullman）。《关于哈希函数效率的一则笔记》。《ACM杂志》（Journal of the ACM (JACM)），1972年，第19卷，第3期，第569 - 575页。

[20] 贝特霍尔德·弗金（Berthold Vöcking）。《不对称性如何有助于负载均衡》。《ACM杂志》（Journal of the ACM (JACM)），2003年，第50卷，第4期，第568 - 589页。

[21] 安德鲁·C·姚（Andrew C Yao）。《均匀哈希是最优的》。《ACM杂志》（Journal of the ACM (JACM)），1985年，第32卷，第3期，第687 - 693页。
