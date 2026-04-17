# 国内 Coding Plan

## 1. Coding Plan

（以下数据更新于 2026-04-17）

### 1.1 阿里云

阿里云 Coding Plan：

[https://common-buy.aliyun.com/coding-plan](https://common-buy.aliyun.com/coding-plan)

[https://bailian.console.aliyun.com/cn-beijing/?tab=coding-plan#/efm/coding-plan-index](https://bailian.console.aliyun.com/cn-beijing/?tab=coding-plan#/efm/coding-plan-index)

### 1.2 腾讯云

腾讯云 Coding Plan：

[https://cloud.tencent.com/act/pro/codingplan](https://cloud.tencent.com/act/pro/codingplan)

### 1.3 火山引擎(字节)

方舟 Coding Plan：

[https://www.volcengine.com/activity/codingplan](https://www.volcengine.com/activity/codingplan)

模型列表：

[https://www.volcengine.com/docs/82379/1330310?lang=zh](https://www.volcengine.com/docs/82379/1330310?lang=zh)

为了节省推理的显存，大多数模型限制上下文最大长度为：200K 或者 256K，最大输入 Max Tokens 限制到 168K 或者 224K，输出 Max Tokens: 32K 。

例如，GLM-5.1：

上下文最大长度：200K，最大输入 Tokens：200K，最大输出 Tokens: 128K 。

### 1.4 清华智谱(GLM)

官网：[https://chatglm.cn](https://chatglm.cn) | [https://www.zhipuai.cn/](https://www.zhipuai.cn/)

GLM Token Plan：

[https://bigmodel.cn/glm-coding](https://bigmodel.cn/glm-coding)

用量说明：

[https://docs.bigmodel.cn/cn/coding-plan/overview#%E7%94%A8%E9%87%8F%E8%AF%B4%E6%98%8E](https://docs.bigmodel.cn/cn/coding-plan/overview#%E7%94%A8%E9%87%8F%E8%AF%B4%E6%98%8E)

### 1.5 MiniMax

MiniMax Token Plan：

[https://platform.minimaxi.com/subscribe/token-plan](https://platform.minimaxi.com/subscribe/token-plan)

套餐价格：

[https://platform.minimaxi.com/docs/guides/pricing-token-plan](https://platform.minimaxi.com/docs/guides/pricing-token-plan)

### 1.6 百度云

百度云 Coding Plan：

[https://console.bce.baidu.com/qianfan/resource/subscribe](https://console.bce.baidu.com/qianfan/resource/subscribe)

### 1.7 京东云

京东云 coding plan：

[https://www.jdcloud.com/cn/products/clawlab](https://www.jdcloud.com/cn/products/clawlab)

活动与公告（购买页面）：[去购买 Coding Plan Pro](https://joybuilder-console.jdcloud.com/system/subscribe/list)

模型广场：

[https://joybuilder-console.jdcloud.com/smart-square/model/list](https://joybuilder-console.jdcloud.com/smart-square/model/list)

为了节省推理的显存，大多数模型限制上下文最大长度为：200K 或者 256K，最大输入 Max Tokens 限制到 168K 或者 224K，输出 Max Tokens: 32K 。

### 1.8 科大讯飞

科大讯飞(星火模型) coding plan：

[https://maas.xfyun.cn/packageSubscription](https://maas.xfyun.cn/packageSubscription)

## 2. Coding Plan 替代方案

### 2.1 阿里云

（入口在 [百炼大模型](https://bailian.console.aliyun.com/cn-beijing) 中的 "工作台" -> "模型用量"，有免费额度的模型后面会有一个 "购买节省计划" 的链接，点进去就能看到）

Qwen3.6 发布 全模型通享低至 4.5 折(包季才能享受 4.5折)

全模型通用抵扣300元（每月抵扣100元），共 135.00元/3个月，每月 45 元。

[https://www.aliyun.com/benefit/scene/ai-discount](https://www.aliyun.com/benefit/scene/ai-discount)

### 2.2 腾讯云(Token Plan)

专门为 龙虾 用户提供的套餐。

个人版–-基础套餐（Standard）：

1亿 Tokens / 99元。

适用于首次体验龙虾，可执行约 200 轮问答。

其他套餐：(Lite) 3500万 Tokens / 39 元、(Pro) 3.2亿 Tokens / 299 元、(Max) 6.5亿 Tokens / 599 元。

购买页面：

[https://cloud.tencent.com/act/pro/tokenplan?Is=home](https://cloud.tencent.com/act/pro/tokenplan?Is=home)

登陆后可访问该页面：

[https://console.cloud.tencent.com/tokenhub/tokenplan?regionId=1](https://console.cloud.tencent.com/tokenhub/tokenplan?regionId=1)

### 2.3 点评

两者各有千秋，阿里云可以使用 Qwen-3.6-Plus、GLM-5.1 等新模型，也支持 Qwen-3.5 系列、GLM-5.1、GLM-5、Kimi-K2.5、MiniMax-M2.5、DeepSeek V3.2 等模型。

腾讯云(Token Plan)的也支持 GLM-5.1，甚至还支持很少见的需要商业授权的 MiniMax-M2.7，除了基本的模型 GLM-5、Kimi-K2.5、MiniMax-M2.5、DeepSeek V3.2 等，还有腾讯自产的 Tencent HY 2.0、Hunyuan-T1 等模型，但没有 Qwen-3.5、Qwen-3.6 以及所有 Qwen 的模型。

两者 Token 用量上不好评估，差不太多，可能腾讯云(Token Plan) 的套餐从 Token 上来说更划算一点，腾讯云多了 MiniMax-M2.7，但少了 Qwen 系列模型。

另外，阿里云是按月抵扣的，当月额度不用完会消失，但没有5小时、周限制。腾讯云(Token Plan)是包干制，没有任何时间限制，也不会有5小时、周限制。
