
# Windows 10 Microsoft Compatibility Telemetry (CompatTelRunner.exe) CPU 占用高的解决办法

## 1. 简介

相信很多人跟我一样总被 `Microsoft Compatibility Telemetry` (`CompatTelRunner.exe`) CPU 高占用，以及硬盘 100% 占用（机械硬盘，如果是 SSD 倒是没事）困扰，`Compatibility Telemetry` 翻译过来就是 “`微软兼容性检测`” 的意思，找了半天终于找到了干掉这个兼容性检测的办法。

## 2. 解决办法

* 禁用服务

控制面板 — 管理工具 — 服务 — 手动（或者禁止，我是手动了，怕以后有需要，如果手动不能解决只能选择禁用） 

    Connected User Experiences and Telemetry（该死的微软还把这个服务改了名字！！，害我找了半天） 
    Diagnostic Policy Service 
    Diagnostic Service Host

* 停止任务计划

控制面板 — 管理工具 — 任务计划

    任务计划程序库 —  Microsoft — Windows — Application Experience — Microsoft Compatibility Appraiser  右键禁止

---------------------

作者：Lawliet丶

来源：[CSDN](http://www.csdn.net)

原文：[https://blog.csdn.net/dKnightL/article/details/69666650](https://blog.csdn.net/dKnightL/article/details/69666650)
