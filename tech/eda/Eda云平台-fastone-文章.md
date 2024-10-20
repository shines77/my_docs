# Eda云平台-fastone

## 1. 文章

- [生信云实证Vol.6：155个GPU！多云场景下的Amber自由能计算](https://fastonetech.com/blog/bio-amber-and-multi-cloud/)

    分子动力学（Molecular Dynamics, MD）模拟实验数据量大，计算周期长，常用软件包括 Amber、NAMD、GROMACS、Schrödinger 等等。GPU的并行处理技术能大大加速计算效率，所以很多MD模拟软件都开始支持GPU。

    `GROMACS` 作为一款开源软件，完全免费，但其成熟版本对于GPU的支持并不理想，教程相对少，对用户的要求比较高。

    `Schrödinger` 是商用软件，功能全面，GPU支持很好，但 License 是按使用核数计算的，价格相对昂贵。

    `Amber` 软件包包括两个部分：AmberTools 和 Amber。

    `AmberTools` 可以在Amber官网免费下载和使用，Tools中包含了Amber绝大部分模块，但不支持PMEMD和GPU加速。

    `Amber` 是收费的，从 `Amber11` 开始支持GPU加速仿真，`Amber18` 开始支持GPU计算自由能，且教程齐全易操作，不限制CORE的使用数量。2020年4月，已经更新到 `Amber20` 版本。

- [ICCAD观后感 | EDA云平台49问](https://fastonetech.com/blog/iccad-eda-49questions/)

  - 1. EDA云平台能够解决什么问题？

    适配EDA工具使用需求。
    大规模算力自动化智能调度。
    海量多云资源提供弹性算力支持。
    总之，让研发人员更专心做设计，帮助IT人员更好地管理资源满足复杂企业场景需求，最终缩短项目周期，提高公司竞争力。

  - 9. 你们和云厂商有什么区别？

    我们是从应用出发，为应用定义的云平台。
    而云厂商主要在IaaS层，距离用户的实际应用还有非常长的距离。
    在云的基础架构和应用之间，需要借助应用优化、多云环境支持等方式来满足用户需求。

- [国内超算发展近40年，终于遇到了一个像样的对手](https://fastonetech.com/blog/superpower/)

  - 计费方式

    预留实例：相当于批发，买定离手。
    主要针对中长期稳定需求，优点是价格整体比较低，缺点是资源必须长期持有，灵活性差。

    按需实例：相当于零售，即买即用。
    针对短期弹性需求，按小时计费，灵活精准，避免浪费，但价格比较高。

    可被抢占实例：相当于秒杀，手快有手慢无。
    价格可高可低波动大，随时可能被抢占，需要有一定的技术实力才能使用。
