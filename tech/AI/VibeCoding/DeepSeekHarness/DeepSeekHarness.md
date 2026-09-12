# DeepSeekHarness

## 1. 简介

DeepSeek Harness（dsh）是由 DeepSeek AI 开发的开源 agent harness（智能体框架）。

它构建于一切皆插件的架构之上，由 Cordis 驱动，其设计参见论文 [A Programming Paradigm for Spatiotemporal Composability。](https://arxiv.org/abs/2608.25512)

文档：[https://deepseek-harness.github.io/deepseek-harness/](https://deepseek-harness.github.io/deepseek-harness)

## 2. 安装

### 2.1 从源安装

```bash
npm install -g @deepseek-ai/dsh
```

### 2.2 从源码安装

```bash
git clone https://github.com/deepseek-ai/deepseek-harness.git
cd deepseek-harness
pnpm install
pnpm run build
pnpm dsh web
```

- `pnpm run build` 会准备仓库产物。
- `pnpm dsh web` 会直接使用这些已构建产物，不会重新构建。

## 3. 运行

```bash
npx @deepseek-ai/dsh web
```
