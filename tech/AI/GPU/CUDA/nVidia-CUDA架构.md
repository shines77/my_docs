# nVidia CUDA 架构

## 1. CUDA 架构

CUDA (Compute Unified Device Architecture) 是一种由 NVIDIA 推出的通用并行计算架构，该架构使 GPU 能够对复杂的计算问题做性能速度优化。

NVIDIA 于 2006 年 11 月在 G80 系列中引入的 Tesla 统一图形和计算架构扩展了 GPU，使其超越了图形领域。通过扩展处理器和存储器分区的数量，其强大的多线程处理器阵列已经成为高效的统一计算平台，同时适用于图形和通用并行计算应用程序。从 G80 系列开始 NVIDIA 加入了对 CUDA 的支持。

具有 Tesla 架构的 GPU 是具有芯片共享存储器的一组 SIMT（单指令多线程）多处理器。它以一个可伸缩的多线程流处理器（Streaming Multiprocessors，SMs）阵列为中心实现了 MIMD（多指令多数据）的异步并行机制，其中每个多处理器包含多个标量处理器（Scalar Processor，SP），为了管理运行各种不同程序的数百个线程，SIMT 架构的多处理器会将各线程映射到一个标量处理器核心，各标量线程使用自己的指令地址和寄存器状态独立执行。

![GPU 架构图](./images/GPU-arch.png)

## 2. nVidia 显卡硬件架构

nVidia 的显卡用 SM、SP 和 Warp 组成。

### 2.1 SM（Streaming Multiprocessor）

SM（多线程流处理器，Streaming Multiprocessor），一个 SM 用多个 SP 和其他资源组成，也叫 GPU 大核，其他资源如：warp scheduler，register，shared memory 等。register 和 shared memory 是 SM 的稀缺资源，CUDA 将这些资源分配给所有驻留在 SM 中的 threads。因此，这些有限的资源就使每个 SM 中 active warps 有非常严格的限制，也就限制了并行能力。如下图所示，是一个 SM 的基本组成，其中每个绿色小块代表一个 SP 。

GPU 的 SM 总数量由具体的架构决定，例如：Ada Lovelace 架构的 RTX 4090 的 SMs/CUs 数量是 128 ，CUDA Cores/流处理器：16384 个，也就是说每个 SM 中有 16384/128 = 128 个 SP 。

### 2.2 SP（Scalar Processor）

SP（标量处理器，Scalar Processor），GPU 最基本的处理单元，也称为 CUDA core ，最后具体的指令和任务都是在 SP 上处理的。GPU 进行并行计算，也就是很多个 SP 同时做处理。

每个 SM 包含的 SP 数量依据 GPU 架构而不同，Fermi 架构 GF100 是 32个，GF10X 是 48 个，Kepler 架构是 192 个，Maxwell 是 128 个。当一个 kernel 启动后，thread 会被分配到很多 SM 中执行。大量的 thread 可能会被分配到不同的 SM ，但是同一个 block 中的 thread 必然在同一个 SM 中并行执行。

![nVidia 显卡硬件架构](./images/nVidia-display-card-arch.jpeg)

### 2.3 Warp

一个 SP 可以执行一个 thread，但是实际上并不是所有的 thread 能够在同一时刻执行。Nvidia 把 32 个 thread 组成一个 warp，warp 是调度和运行的基本单元。warp 中所有 threads 并行的执行相同的指令。warp 由 SM 的硬件 warp scheduler 负责调度，一个 SM 同一个时刻可以执行多个 warp，这取决于 warp scheduler 的数量。目前每个 warp 包含 32 个 threads（nVidia 保留修改数量的权利）。

## 3. CUDA 软件架构

nVidia 的 CUDA 软件架构主要用 Kernel、Grid、Block 组成。

### 3.1 Kernel

具体到我们如何调用 GPU 上的线程实现我们的算法，则是通过 Kernel 实现的。在 GPU 上调用的函数成为 CUDA 核函数（Kernel function），核函数会被 GPU 上的多个线程执行。我们可以通过如下方式来定义一个 kernel：

```cpp
kernel_function<<<grid, block>>>(param1, param2, param3....);
```

### 3.2 Grid

Grid 是由一个单独的 Kernel 启动的，所有线程组成一个 Grid，Grid 中所有线程共享 global memory。Grid 由很多 Block 组成，可以是一维二维或三维。

### 3.3 Block

一个 Grid 由许多 Block 组成，Block 由许多 thread 组成，同样可以有一维、二维或者三维。Block 内部的多个 thread 可以同步（synchronize），可访问共享内存（share memory）。

如下图所示为 kernel、grid、block 的组织方式以及对应的硬件部分。

![kernel、grid、block 的组织方式](./images/CUDA-software-arch.png)


## x. 参考文章

- [异构计算--CUDA架构](https://blog.csdn.net/qq_44924694/article/details/126202388)

- [CUDA并行架构](https://blog.csdn.net/qq_41636999/article/details/142392850)
