# OPSX 工作流

> 欢迎在 [Discord](https://discord.gg/YctCnvvshC) 提供反馈。

## 这是什么？

OPSX 现在是 OpenSpec 的标准工作流。

它是一个**流畅、迭代式的工作流**，用于 OpenSpec 的改进。不再有僵化的阶段——只有你可以随时执行的动作。

## 为什么存在这个

传统的 OpenSpec 工作流能工作，但它是**锁定的**：

- **指令是硬编码的**——隐藏在 TypeScript 中，你无法更改它们
- **全有或全无**——一个大命令创建所有内容，无法测试单个部分
- **固定结构**——每个人都用相同的工作流，没有自定义
- **黑盒**——当 AI 输出不好时，你无法调整提示词

**OPSX 将其打开。** 现在任何人都可以：

1. **尝试指令**——编辑模板，看看 AI 是否做得更好
2. **细粒度测试**——独立验证每个 Artifact 的指令
3. **自定义工作流**——定义你自己的 Artifact 和依赖关系
4. **快速迭代**——更改模板，立即测试，无需重建

```text
传统工作流:                          OPSX:
┌────────────────────────┐           ┌────────────────────────┐
│   在包中硬编码         │           │  schema.yaml           │◄── 你编辑这个
│   (无法更改)           │           │  templates/*.md        │◄── 或这个
│        ↓               │           │        ↓               │
│   等待新版本           │           │    即时生效            │
│        ↓               │           │        ↓               │
│   希望它更好           │           │    自己测试            │
└────────────────────────┘           └────────────────────────┘
```

**这是给每个人的：**

- **团队**——创建符合您实际工作方式的工作流
- **高级用户**——调整提示词，为你的代码库获得更好的 AI 输出
- **OpenSpec 贡献者**——无需发布即可尝试新方法

我们仍在学习什么是最有效的。OPSX 让我们一起学习。

## 用户体验

**线性工作流的问题是：**
你"处于 planning 阶段"，然后"处于 implementation 阶段"，然后"完成"。但真实的工作并不是这样运作的。你实现了一些东西，意识到你的设计是错的，需要更新规范，继续实现。线性阶段与工作的实际方式相矛盾。

**OPSX 方法：**

- **Actions，而不是阶段**——创建、实现、更新、归档——随时执行其中任何一个
- **依赖关系是推动者**——它们展示了什么是可能的，而不是接下来需要什么

```text
  proposal ──→ specs ──→ design ──→ tasks ──→ implement
```

## 设置

```bash
# 确保你已安装 openspec —— skills 会自动生成
openspec init
```

这在 `.claude/skills/`（或等效目录）中创建技能，AI 编码助手会自动检测。

在设置过程中，系统会提示你创建**项目配置**（`openspec/config.yaml`）。这是可选的，但推荐。

## 项目配置

项目配置允许你设置默认值，并将项目特定的上下文注入到所有工件中。

### 创建配置

配置在 `openspec init` 期间创建的，或手动创建的：

```yaml
# openspec/config.yaml
schema: spec-driven

context: |
  Tech stack: TypeScript, React, Node.js
  API conventions: RESTful, JSON responses
  Testing: Vitest for unit tests, Playwright for e2e
  Style: ESLint with Prettier, strict TypeScript

rules:
  proposal:
    - Include rollback plan
    - Identify affected teams
  specs:
    - Use Given/When/Then format for scenarios
  design:
    - Include sequence diagrams for complex flows
```

### 配置字段

| 字段 | 类型 | 描述 |
|-------|------|-------------|
| `schema` | string | 新更改的默认模式 (例如, `spec-driven`) |
| `context` | string | 项目上下文注入到所有工件指令中 |
| `rules` | object | 每个工件的规则，由工件 ID 键入 |

### 工作原理

**模式优先级** (从高到低):

1. CLI 标志 (`--schema <name>`)
2. 更改元数据 (更改目录中的 `.openspec.yaml`)
3. 项目配置 (`openspec/config.yaml`)
4. 默认 (`spec-driven`)

**上下文注入:**

- 上下文被添加到每个 Artifact 的指令之前
- 包装在 `<context>...</context>` 标签中
- 帮助 AI 理解你项目的约定

**规则注入:**

- 规则仅针对匹配的 Artifact 注入
- 包装在 `<rules>...</rules>` 标签中
- 出现在上下文之后，模板之前

### 按模式划分的 Artifact ID

**spec-driven** (默认):

- `proposal` — 更改提案
- `specs` — 产品规范
- `design` — 技术设计
- `tasks` — 实施任务

### 配置验证

- `rules` 中未知的 Artifact ID 会生成警告
- Schema 名称会根据可用 Schema 进行验证
- 上下文有 50KB 大小限制
- 无效的 YAML 会报告行号

### 故障排除

**"rules 中未知的 Artifact ID: X"**

- 检查 Artifact ID 是否匹配你的 Schema (参见上面的列表)
- 运行 `openspec schemas --json` 查看每个 schemas 的 Artifact ID

**配置未被应用:**

- 确保文件位于 `openspec/config.yaml` (而不是 `.yml`)
- 使用验证器检查 YAML 语法
- 配置更改立即生效 (无需重启)

**上下文太大:**

- 上下文限制为 50KB
- 改为总结或链接到外部文档

## 命令

| 命令 | 作用 |
|---------|--------------|
| `/opsx:explore` | 思考想法、研究问题、明确需求 |
| `/opsx:new` | 开始一个新的变更 |
| `/opsx:continue` | 创建下一个 Artifact (基于已准备好的内容) |
| `/opsx:ff` | Fast-forward —— 一次性创建所有规划的 Artifact |
| `/opsx:apply` | 实施任务，根据需要更新 Artifact |
| `/opsx:sync` | 将增量规格(delta spec)同步到主规格(main spec) (可选——如果需要，会提示归档) |
| `/opsx:archive` | 完成后归档 |

## 用法

### 探索一个想法

```text
/opsx:explore
```

思考想法、研究问题，比较选项。不需要结构——只需一个有思想的合作伙伴。当见解具体化时，请过渡到 `/opsx:new` 或 `/opsx:ff`。

### 开始新变更

```text
/opsx:new
```

系统会询问你想要构建什么以及使用哪个工作流模式。

### 创建 Artifact

```text
/opsx:continue
```

根据依赖关系显示准备创建的内容，然后创建一个 Artifact。反复使用以逐步建立您的更改。

```text
/opsx:ff add-dark-mode
```

一次性创建所有规划的 Artifact。当您对正在构建的内容有清晰的了解时使用。

### 实现

```text
/opsx:apply
```

执行任务，边做边检查。如果你要同时处理多个更改，可以运行 `/opsx:apply <name>`；否则它应该从对话中推断，如果无法确定则提示你选择。

### 完成

```text
/opsx:archive   # 完成时移动到归档 (如果需要则提示 sync specs)
```

## 何时更新 vs 重新开始

你始终可以在 implementation 之前编辑你的提案或规范。但什么时候细化会变成"这是不同的工作"呢？

### 提案捕获了什么

提案定义了三件事：

1. **意图** —— 你要解决什么问题？
2. **范围** —— 什么是界内/界外？
3. **方法** —— 你将如何解决这个问题？

问题是：哪个改变了，改变了多少？

### 在以下情况下更新现有的更改

**相同意图，细化执行**

- 你发现了你没有考虑到的边缘情况
- 方法需要调整，但目标不变
- 实现揭示设计略有偏差

**范围缩小**

- 你意识到完整的范围太大了  ，想要先交付 MVP
- "添加深色模式" → "添加深色模式切换 (v2 中的系统偏好设置)"

**学习驱动的修正**

- 代码库的结构与您的想法不同
- 依赖项未按预期工作
- "使用 CSS 变量" → "使用 Tailwind 的 dark: 前缀代替"

### 在以下情况下开始新的更改

**意图根本改变**

- 现在问题本身不同了
- "添加深色模式" → "添加具有自定义颜色、字体、间距的综合主题系统"

**范围爆炸**

- 变化如此之大，本质上是不同的工作
- 更新后原提案将无法识别
- "修复登录 bug" → "重写认证系统"

**原始内容已完成**

- 原始更改可以标记为"完成"
- 新工作是独立的，而不是改进
- 完成"添加深色模式 MVP" → 归档 → 新更改"增强深色模式"

### 启发式方法

```text
                        ┌─────────────────────────────────────┐
                        │     这是相同的工作吗？              │
                        └──────────────┬──────────────────────┘
                                       │
                    ┌──────────────────┼──────────────────┐
                    │                  │                  │
                    ▼                  ▼                  ▼
                相同意图？          >50% 重叠？       原始内容可以在
                相同问题？          相同范围？    没有这些更改的情况下
                    │                  │             被"完成"吗？
                    │                  │                  │
           ┌────────┴────────┐  ┌──────┴──────┐   ┌───────┴───────┐
           │                 │  │             │   │               │
          YES               NO YES           NO  NO              YES
           │                 │  │             │   │               │
           ▼                 ▼  ▼             ▼   ▼               ▼
        UPDATE            NEW  UPDATE       NEW  UPDATE          NEW
```

| 测试 | 更新 | 新更改 |
|------|--------|------------|
| **身份** | "同样的东西，但细化了" | "不一样的工作" |
| **范围重叠** | >50% 重叠 | <50% 重叠 |
| **完成** | 不进行改变就无法“完成” | 可以完成原始内容，新工作独立存在 |
| **故事** | 更新链讲述连贯的故事 | 补丁只会让人困惑而不是澄清 |

### 原则

> **更新保留上下文。新的更改提供清晰度。**
>
> 当您的思维历史很有价值时，请选择更新。当重新开始比修补更清晰时选择新的更改。

把它想象成 git 分支：

- 在开发相同功能时，持续提交
- 当真正的新工作出现时，开始一个新的分支
- 有时会合并部分功能，并在第二阶段重新开始

## 有什么不同？

| | 传统工作流 (`/openspec:proposal`) | OPSX (`/opsx:*`) |
|---|---|---|
| **结构** | 一个大的提案文档 | 带有依赖关系的离散 Artifact |
| **工作流** | 线性阶段：plan → implement → archive | 流畅的行动——随时做任何事 |
| **迭代** | 难以返回 | 在学习过程中更新 Artifact |
| **自定义** | 固定结构 | 模式驱动（定义你自己的 Artifact） |

**核心见解：** 工作不是线性的。OPSX 不再假装它是。

## 架构深度解析

本节解释 OPSX 的底层工作原理以及它与传统工作流的比较。

### 理念：Phases vs Actions

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                         传统工作流                                          │
│                    （阶段锁定，全有或全无）                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌──────────────┐      ┌──────────────┐      ┌──────────────┐              │
│   │   规划阶段   │ ───► │  实现阶段    │ ───► │   归档阶段   │              │
│   └──────────────┘      └──────────────┘      └──────────────┘              │
│         │                     │                     │                       │
│         ▼                     ▼                     ▼                       │
│   /openspec:proposal   /openspec:apply      /openspec:archive               │
│                                                                             │
│   • 一次性创建所有 Artifact                                                 │
│   • 在实现期间无法返回更新规范                                              │
│   • 阶段门强制线性推进                                                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────┐
│                            OPSX 工作流                                      │
│                      （流畅的 Actions，迭代式）                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│              ┌──────────────────────────────────────────────┐               │
│              │           Actions（不是阶段）                │               │
│              │                                              │               │
│              │   new ◄──► continue ◄──► apply ◄──► archive  │               │
│              │    │          │           │           │      │               │
│              │    └──────────┴───────────┴───────────┘      │               │
│              │              任意顺序                        │               │
│              └──────────────────────────────────────────────┘               │
│                                                                             │
│   • 一次创建一个 Artifact 或快速推进 (fast-forward)                         │
│   • 在实现期间更新规范/设计/任务                                            │
│   • 依赖启用进度，阶段不存在                                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 组件架构

**传统工作流**：使用 TypeScript 中的硬编码模板。

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                      传统工作流组件                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   硬编码模板（TypeScript 字符串）                                           │
│                    │                                                        │
│                    ▼                                                        │
│   配置器（18+ 类，每个编辑器一个）                                          │
│                    │                                                        │
│                    ▼                                                        │
│   生成的命令文件（.claude/commands/openspec/*.md）                          │
│                                                                             │
│   • 固定结构，无 Artifact 感知                                              │
│   • 更改需要代码修改 + 重建                                                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**OPSX**：使用外部模式和依赖图引擎。

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                         OPSX 组件                                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Schema 定义（YAML）                                                       │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  name: spec-driven                                                  │   │
│   │  artifacts:                                                         │   │
│   │    - id: proposal                                                   │   │
│   │      generates: proposal.md                                         │   │
│   │      requires: []              ◄── 依赖                             │   │
│   │    - id: specs                                                      │   │
│   │      generates: specs/**/*.md  ◄── Glob 模式                        │   │
│   │      requires: [proposal]      ◄── 在 proposal 之后启用             │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                    │                                                        │
│                    ▼                                                        │
│    Artifact 图引擎                                                          │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  • 拓扑排序（依赖排序）                                             │   │
│   │  • 状态检测（文件系统存在性）                                       │   │
│   │  • 丰富指令生成（模板 + 上下文）                                    │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                    │                                                        │
│                    ▼                                                        │
│   Skill 文件（.claude/skills/openspec-*/SKILL.md）                          │
│                                                                             │
│   • 跨编辑器兼容（Claude Code, Cursor, Windsurf）                           │
│   • Skills 查询 CLI 获取结构化数据                                          │
│   • 完全可通过 schema 文件自定义                                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 依赖图模型

 Artifact 形成有向无环图（DAG）。依赖关系是**推动者**，而不是门：

```text
                              proposal
                             (根节点)
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
                    ▼                           ▼
                 specs                       design
              (requires:                  (requires:
               proposal)                   proposal)
                    │                           │
                    └─────────────┬─────────────┘
                                  │
                                  ▼
                               tasks
                           (requires:
                           specs, design)
                                  │
                                  ▼
                          ┌──────────────┐
                          │ APPLY 阶段   │
                          │ (requires:   │
                          │  tasks)      │
                          └──────────────┘
```

**状态转换：**

```text
   BLOCKED ────────────────► READY ────────────────► DONE
      │                        │                       │
   缺少                      所有依赖                 文件存在
   依赖                      都已 DONE               在文件系统上
```

### 信息流

**传统工作流** —— 代理接收静态指令：

```text
  用户: "/openspec:proposal"
           │
           ▼
  ┌─────────────────────────────────────────┐
  │  静态指令:                              │
  │  • Create proposal.md                   │
  │  • Create tasks.md                      │
  │  • Create design.md                     │
  │  • Create specs/<capability>/spec.md    │
  │                                         │
  │  不了解存在什么或                       │
  │   Artifact 之间的依赖                   │
  └─────────────────────────────────────────┘
           │
           ▼
  代理一次性创建所有 Artifact
```

**OPSX** —— 代理查询丰富的上下文：

```text
  用户: "/opsx:continue"
           │
           ▼
  ┌──────────────────────────────────────────────────────────────────────────┐
  │  步骤 1: 查询当前状态                                                    │
  │  ┌────────────────────────────────────────────────────────────────────┐  │
  │  │  $ openspec status --change "add-auth" --json                      │  │
  │  │                                                                    │  │
  │  │  {                                                                 │  │
  │  │    "artifacts": [                                                  │  │
  │  │      {"id": "proposal", "status": "done"},                         │  │
  │  │      {"id": "specs", "status": "ready"},      ◄── 第一个就绪       │  │
  │  │      {"id": "design", "status": "ready"},                          │  │
  │  │      {"id": "tasks", "status": "blocked", "missingDeps": ["specs"]}│  │
  │  │    ]                                                               │  │
  │  │  }                                                                 │  │
  │  └────────────────────────────────────────────────────────────────────┘  │
  │                                                                          │
  │  步骤 2: 获取就绪 Artifact 的丰富指令                                    │
  │  ┌────────────────────────────────────────────────────────────────────┐  │
  │  │  $ openspec instructions specs --change "add-auth" --json          │  │
  │  │                                                                    │  │
  │  │  {                                                                 │  │
  │  │    "template": "# Specification\n\n## ADDED Requirements...",      │  │
  │  │    "dependencies": [{"id": "proposal", "path": "...", "done": true}│  │
  │  │    "unlocks": ["tasks"]                                            │  │
  │  │  }                                                                 │  │
  │  └────────────────────────────────────────────────────────────────────┘  │
  │                                                                          │
  │  步骤 3: 读取依赖 → 创建一个 Artifact → 显示解锁了什么                   │
  └──────────────────────────────────────────────────────────────────────────┘
```

### 迭代模型

**传统工作流** —— 难以迭代：

```text
  ┌─────────┐     ┌─────────┐     ┌─────────┐
  │/proposal│ ──► │ /apply  │ ──► │/archive │
  └─────────┘     └─────────┘     └─────────┘
       │               │
       │               ├── "等等，设计是错的"
       │               │
       │               ├── 选项:
       │               │   • 手动编辑文件 (破坏上下文)
       │               │   • 放弃并重新开始
       │               │   • 继续推进，稍后修复
       │               │
       │               └── 没有官方的"返回"机制
       │
       └── 一次性创建所有 Artifact
```

**OPSX** —— 自然迭代：

```text
  /opsx:new ───► /opsx:continue ───► /opsx:apply ───► /opsx:archive
      │                │                  │
      │                │                  ├── "设计是错的"
      │                │                  │
      │                │                  ▼
      │                │            只需编辑 design.md
      │                │            然后继续！
      │                │                  │
      │                │                  ▼
      │                │         /opsx:apply 从
      │                │         你离开的地方继续
      │                │
      │                └── 创建一个 Artifact，显示解锁了什么
      │
      └── 搭建变更，等待方向
```

### 自定义模式

使用模式管理命令创建自定义工作流：

```bash
# 从头创建新模式（交互式）
openspec schema init my-workflow

# 或 fork 现有模式作为起点
openspec schema fork spec-driven my-workflow

# 验证你的模式结构
openspec schema validate my-workflow

# 查看模式从哪里解析（对调试有用）
openspec schema which my-workflow
```

Schemas 存储在 `openspec/schemas/`（项目本地，版本控制）或 `~/.local/share/openspec/schemas/`（用户全局）中。

**Schema 结构：**

```text
openspec/schemas/research-first/
├── schema.yaml
└── templates/
    ├── research.md
    ├── proposal.md
    └── tasks.md
```

**schema.yaml 示例：**

```yaml
name: research-first
artifacts:
  - id: research        # 在 proposal 之前添加
    generates: research.md
    requires: []

  - id: proposal
    generates: proposal.md
    requires: [research]  # 现在依赖 research

  - id: tasks
    generates: tasks.md
    requires: [proposal]
```

**依赖关系图：**

```text
   research ──► proposal ──► tasks
```

### 总结

| 方面 | 传统 | OPSX |
|--------|----------|------|
| **模板** | 硬编码 TypeScript | 外部 YAML + Markdown |
| **依赖关系** | 无（一次性全部） | 具有拓扑排序的 DAG  |
| **状态** | 基于阶段的心理模型  | 文件系统存在性 |
| **自定义** | 编辑源码，重建 | 创建 schema.yaml |
| **迭代** | 阶段锁定 | 流畅，编辑任何内容 |
| **编辑器支持** | 18+ 个配置器类 | 单一 skills 目录 |

## Schemas (模式)

Schemas 定义了存在哪些 Artifact 及其依赖关系。目前可用：

- **spec-driven** (默认): proposal → specs → design → tasks

```bash
# 列出可用模式
openspec schemas

# 查看所有模式及其解析源
openspec schema which --all

# 交互式创建新模式
openspec schema init my-workflow

# Fork 现有模式进行自定义
openspec schema fork spec-driven my-workflow

# 使用前验证模式结构
openspec schema validate my-workflow
```

## 温馨提示

- 在做出改变之前，请使用 `/opsx:explore` 仔细思考一个想法。
- 当你知道自己想要什么是用 `/opsx:ff`，当探索时用 `/opsx:continue`。
- 在 `/opsx:apply` 期间，如果出现问题——修复 Artifact，然后继续。
- 任务通过 `tasks.md` 中的复选框跟踪进度。
- 随时检查状态：`openspec status --change "name"`

## 反馈意见

这很粗糙。这是有意为之的——我们正在学习什么是有效的。

发现 bug？有想法？加入我们的 [Discord](https://discord.gg/YctCnvvshC) 或在 [GitHub](https://github.com/Fission-AI/openspec/issues) 上开启 issue。
