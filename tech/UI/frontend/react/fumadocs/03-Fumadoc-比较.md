# Fumadocs Framework: 比较

URL: /docs/ui/comparisons
Source: https://raw.githubusercontent.com/fuma-nama/fumadocs/refs/heads/main/apps/docs/content/docs/ui/comparisons.mdx

Fumadocs 与其他现有框架有何不同？
        
## Nextra

Fumadocs 深受 Nextra 的启发。例如，路由约定。这就是为什么
Fumadocs 中也存在 `meta.json`。

Nextra 比 Fumadocs 更具意见性，作为副作用，你必须
手动配置事物，而不像简单编辑配置文件那样。

如果您希望对一切拥有更多控制权，例如
将其添加到现有代码库或实现高级路由，
Fumadocs 效果很好。

### 功能表

| 功能          | Fumadocs | Nextra             |
| ----------- | -------- | ------------------ |
| 静态生成        | 是        | 是                  |
| 缓存          | 是        | 是                  |
| 浅色/深色模式     | 是        | 是                  |
| 语法高亮        | 是        | 是                  |
| 目录          | 是        | 是                  |
| 全文搜索        | 是        | 是                  |
| i18n        | 是        | 是                  |
| 最后 Git 编辑时间 | 是        | 是                  |
| 页面图标        | 是        | 是，通过 `_meta.js` 文件 |
| RSC         | 是        | 是                  |
| 远程源         | 是        | 是                  |
| SEO         | 通过元数据    | 是                  |
| 内置组件        | 是        | 是                  |
| RTL 布局      | 是        | 是                  |

### 额外功能

通过第三方库（如 [TypeDoc](https://typedoc.org)）支持的功能将不列于此。

| 功能                  | Fumadocs | Nextra |
| ------------------- | -------- | ------ |
| OpenAPI 集成          | 是        | 否      |
| TypeScript 文档生成     | 是        | 否      |
| TypeScript Twoslash | 是        | 是      |

## Mintlify

Mintlify 是一个文档服务，与 Fumadocs 相比，它提供免费层级，但并非完全免费且开源。

Fumadocs 没有 Mintlify 那么强大，例如 Mintlify 的 OpenAPI 集成。
作为 Fumadocs 的创建者，如果您对当前构建文档的方式满意，我不推荐从 Mintlify 切换到 Fumadocs。
然而，我相信 Fumadocs 是所有希望拥有优雅文档的 React.js 开发者的合适工具。

## Docusaurus

Docusaurus 是一个基于 React.js 的强大框架。它通过插件和自定义主题提供许多酷炫
功能。

### 较低复杂度

由于 Fumadocs 旨在与 React 框架集成，您可能需要更多 React.js 知识才能入门。
作为回报，Fumadocs 具有更好的自定义性。

对于简单文档，如果您不需要任何特定于框架的功能，Docusaurus 可能是一个更好的选择。

### 插件

您可以轻松通过插件实现许多功能，其生态系统确实更大，由许多贡献者维护。

相比之下，Fumadocs 的灵活性允许您自行实现它们，可能需要更长时间来调整到您的满意程度。
