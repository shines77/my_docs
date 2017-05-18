
华为 RH2288 V3 服务器独立显卡安装指南
--------------------------------------

# 1. 概要 #

服务器型号： `华为 RH2288 V3`。

显卡：`PCIe` 接口的 `Nvdia 10xx` 显卡。

目标：研究在服务器 `华为 RH2288 V3` 是否支持该显卡，如果支持，该如何安装。

# 2. PCIe插槽 #

本章节及 `3.1` 小节内容均摘自：[华为FusionServer RH2288 V3技术白皮书](https://wenku.baidu.com/view/eced52b1a26925c52dc5bf1a.html)

## 2.1 逻辑结构图 ##

![华为 RH2288 V3 逻辑结构图](./images/RH2288_v3_logic_structure.png)

## 2.2 PCIe 插槽分布 ##

![华为 RH2288 V3 PCIe 插槽分布](./images/RH2288_v3_PCIe_slots.png)

## 2.3 PCIe 插槽和 CPU 的关系图 ##

![PCIe 插槽和 CPU 的关系图](./images/RH2288_v3_PCIe_slots_table.png)

# 3. 物理结构 #

## 3.1 服务器的部件结构图 ##

![服务器的部件结构图](./images/RH2288_v3_equipments.png)

## 3.2 服务器的内部结构图 ##

本小节内容摘自：[FusionServer全升级 华为RH2288 V3拆解](http://server.it168.com/a2015/0114/1698/000001698641_all.shtml)

![服务器的内部结构图](./images/RH2288_v3_hardware_structure.png)

# 4. PCIe 插槽实物 #

## 4.1 京东链接 ##

链接：[华为（HUAWEI) PCIE卡及后置硬盘组件 用于2288V3/RH2288HV3 2*16X PCIE卡组件](http://item.jd.com/1782164226.html#crumb-wrap)

![服务器的内部结构图](./images/RH2288_v3_PCIe_2x16.png)

## 4.2 型号 ##

![服务器的内部结构图](./images/RH2288_v3_PCIe_2x16_model.png)

# 5. 实际案例 #

本小节内容参考自：[华为RH2288H V3服务器如何安装英伟达K4000独立显卡使用案例](http://www.ict18.com/320.html)

**问题描述**

现场 `RH2288H V3` 服务器交付时根据客户使用需求给客户在服务器的 `Riser` 卡配置了一块 `Nvidia K4000` 独立显卡，装好系统后发现不能使用独显功能。

**处理方案如下**

1） 将服务器 `BIOS` 设置显卡项选择由 `板载显卡` 切换为 `外接显卡`。

![BIOS设置为外界显卡](./images/BIOS_videocard_selected.png)

2） 独立显卡需要一根 `8PIN` 的电源线独立供电。

3） `Nvidia K4000` 显卡占用资源太多，启动系统时有冲突，需要关闭 `BIOS` 中串口重定向功能释放资源。

![BIOS设置为外界显卡](./images/BIOS_close_serial_redirect.png)

# 6. 参考文献 #

1. [华为FusionServer RH2288 V3技术白皮书](https://wenku.baidu.com/view/eced52b1a26925c52dc5bf1a.html)

2. [FusionServer全升级 华为RH2288 V3拆解](http://server.it168.com/a2015/0114/1698/000001698641_all.shtml)

3. [华为（HUAWEI) PCIE卡及后置硬盘组件 用于2288V3/RH2288HV3 2*16X PCIE卡组件](http://item.jd.com/1782164226.html#crumb-wrap)

4. [华为RH2288H V3服务器如何安装英伟达K4000独立显卡使用案例](http://www.ict18.com/320.html)
