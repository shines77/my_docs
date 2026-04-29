# Fumadoc Framework 入门教程

## 简介

Fumadocs (Foo-ma docs) 是一个 **文档框架**，旨在快速、灵活，并与您的 React 框架无缝集成。

Fumadocs 包含不同的部分：

- **Fumadocs Core**：处理大部分逻辑，包括文档搜索、内容源适配器和 Markdown 扩展。
- **Fumadocs UI**：Fumadocs 的默认主题为文档站点提供美观的外观和交互式组件。
- **Content Source**：您的内容来源，可以是 CMS 或本地数据层，如 [Fumadocs MDX](/docs/mdx)（官方内容源）。
- **Fumadocs CLI**：一个命令行工具，用于安装 UI 组件和自动化任务，有助于自定义布局。

>
> 想了解更多？
> 阅读我们深入的 [什么是 Fumadocs](https://fumadocs.cndocs.org/docs/ui/what-is-fumadocs) 介绍。
>

- GitHub：[https://github.com/fuma-nama/fumadocs](https://github.com/fuma-nama/fumadocs)
- 官方文档：[https://www.fumadocs.dev/docs](https://www.fumadocs.dev/docs)
- 中文文档：[https://fumadocs.cndocs.org/docs/ui](https://fumadocs.cndocs.org/docs/ui)

## 术语

**Markdown/MDX：** Markdown 是一种用于创建格式化文本的标记语言。Fumadocs 原生支持 Markdown 和 MDX（Markdown 的超集）。

**[Bun](https://bun.sh)：** 一个 JavaScript 运行时，我们用它来运行脚本。

对 React.js 的一些基本知识将有助于进一步自定义。

## 自动安装

需要 Node.js 20 的最低版本。

npm 安装：

```bash
npm create fumadocs-app
```

pnpm 安装：

```bash
pnpm create fumadocs-app
```

yarm, bun 的安装方法类似。

它将询问您要使用的内置模板：

* **React.js 框架**：Next.js、Waku、React Router、Tanstack Start。
* **内容源**：Fumadocs MDX。

一个新的 fumadocs 应用将被初始化。现在您可以开始开发！

>
> 从现有代码库开始？
> 您可以跟随 [手动安装](https://fumadocs.cndocs.org/docs/ui/manual-installation) 指南开始。
>

## 享受过程！

在 docs 文件夹中创建您的第一个 MDX 文件。

`content/docs/index.mdx`：

```mdx
---
title: Hello World
---

## Yo what's up
```

在开发模式下运行应用，并查看 [http://localhost:3000/docs。](http://localhost:3000/docs。)

npm 方式：

```bash
npm run dev
```

pnmp 方式：

```bash
pnpm run dev
```

## 了解更多

新来这里？别担心，我们欢迎您的问题。

如果您发现任何令人困惑的内容，请在 [Github Discussion](https://github.com/fuma-nama/fumadocs/discussions) 上提供反馈！

## 编写内容

对于编写文档，请确保阅读：

- [Markdown](https://fumadocs.cndocs.org/docs/ui/markdown)：Fumadocs 为编写内容提供了一些额外功能。
- [导航](https://fumadocs.cndocs.org/docs/ui/navigation)：了解如何自定义导航链接和侧边栏项。
- [页面 Slug 和页面树](https://fumadocs.cndocs.org/docs/ui/page-conventions)：页面 Slug 和页面树
- [组件](https://fumadocs.cndocs.org/docs/ui/components)：查看所有可用的组件来增强您的文档

## 特殊需求

- [配置静态导出](https://fumadocs.cndocs.org/docs/ui/static-export)：了解如何在您的文档上启用静态导出
- [国际化](https://fumadocs.cndocs.org/docs/ui/internationalization)：了解如何启用 i18n
- [颜色主题](https://fumadocs.cndocs.org/docs/ui/theme)：为 Fumadocs UI 添加主题
- [布局](https://fumadocs.cndocs.org/docs/ui/components)：自定义您的 Fumadocs UI 布局

## 引用

- [Fumadocs - 快速开始](https://fumadocs.cndocs.org/docs/ui)
