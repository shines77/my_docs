# ClaudeCode 使用技巧

## 1. Superpowers

官网：[https://github.com/obra/superpowers](https://github.com/obra/superpowers)

### 1.1 安装 Superpowers

在 Claude Code 中，先注册 marketplace ：

```bash
/plugin marketplace add obra/superpowers-marketplace
```

然后在 marketplace 里安装 plugin ：

```bash
/plugin install superpowers@superpowers-marketplace
```

Note: 不同的平台的安装有所不同，Claude Code 有一个内建的 plugin 系统，而 Codex，OpenCode 则需求手动安装。

恢复官方的 markerplace ：

```bash
/plugin marketplace add anthropics/claude-plugins-official
/plugin marketplace add anthropics/skills
```

```bash
/plugin install pdf@anthropic-agent-skills
/plugin install pptx@anthropic-agent-skills
/plugin install docx@anthropic-agent-skills
/plugin install theme-factory@anthropic-agent-skills
/plugin install skill-creator@anthropic-agent-skills
/plugin install frontend-design@anthropic-agent-skills
/plugin install mcp-builder@anthropic-agent-skills
```

SKILL.md 文件的格式说明：[https://agentskills.io/specification](https://agentskills.io/specification) 。

### 1.2 验证安装

开始一个 Claude session 向 LLM 提问来触发 skill (例如："help me plan this feature" or "let's debug this issue"). Claude 将自动调用 superpowers skill.

## 2. 基本工作流

1. **brainstorming**: (头脑风暴) - 在编写代码前进行。通过提问来细化初步想法，探索替代方案，分部分展示设计以供验证。保存设计文档。

2. **using-git-worktrees**: (使用Git工作树) - 在设计批准后激活。在新分支上创建隔离的工作区，运行项目设置，验证测试基线是否干净。

3. **writing-plans**: (编写计划) - 根据已批准的设计启动。将工作分解为小块任务（每项任务2-5分钟）。每项任务都有确切的文件路径、完整的代码和验证步骤。

4. **subagent-driven-development** or **executing-plans**: (子代理驱动开发或执行计划) - 根据计划激活。为每个任务分派新的子代理，并进行两阶段审查（规范符合性审查，然后是代码质量审查），或通过人工检查点分批执行。

5. **test-driven-development**: (测试驱动开发) - 在实现阶段启动。遵循 RED-GREEN-REFACTOR 原则：编写失败的测试，观察其失败，编写最少的代码，观察其通过，然后提交。删除在测试之前编写的代码。

6. **requesting-code-review**: (请求代码审查) - 在任务之间激活。根据计划进行审查，按严重程度报告问题。关键问题会阻碍进度。

7. **finishing-a-development-branch**: (完成一个开发分支) - 任务完成后激活。验证测试，显示选项（合并/拉取请求/保留/丢弃），清理工作树。

代理人在执行任何任务之前都会检查相关技能。这是强制性的工作流程，而非建议。

## 3. The Basic Workflow

1. **brainstorming** - Activates before writing code. Refines rough ideas through questions, explores alternatives, presents design in sections for validation. Saves design document.

2. **using-git-worktrees** - Activates after design approval. Creates isolated workspace on new branch, runs project setup, verifies clean test baseline.

3. **writing-plans** - Activates with approved design. Breaks work into bite-sized tasks (2-5 minutes each). Every task has exact file paths, complete code, verification steps.

4. **subagent-driven-development** or **executing-plans** - Activates with plan. Dispatches fresh subagent per task with two-stage review (spec compliance, then code quality), or executes in batches with human checkpoints.

5. **test-driven-development** - Activates during implementation. Enforces RED-GREEN-REFACTOR: write failing test, watch it fail, write minimal code, watch it pass, commit. Deletes code written before tests.

6. **requesting-code-review** - Activates between tasks. Reviews against plan, reports issues by severity. Critical issues block progress.

7. **finishing-a-development-branch** - Activates when tasks complete. Verifies tests, presents options (merge/PR/keep/discard), cleans up worktree.

The agent checks for relevant skills before any task. Mandatory workflows, not suggestions.

## 4. 内部构造

### Skills 库

**Testing (测试)**

- **test-driven-development** - RED-GREEN-REFACTOR 循环（包括测试反模式参考）

**Debugging (调试)**

- **systematic-debugging** - 四阶段根本原因分析流程（包括根本原因追踪、深度防御、基于条件的等待技术）
- **verification-before-completion** - 确保问题确实已得到解决

**Collaboration (合作)**

- **brainstorming** - 苏格拉底式设计优化
- **writing-plans** - 详细的实施计划
- **executing-plans** - 带检查点的批量执行
- **dispatching-parallel-agents** - 并行子代理工作流
- **requesting-code-review** - 预审检查清单
- **receiving-code-review** - 回应反馈
- **using-git-worktrees** - 并行开发分支
- **finishing-a-development-branch** - 合并/拉取请求决策工作流程
- **subagent-driven-development** - 通过两阶段评审（规范符合性，然后是代码质量）实现快速迭代

**Meta (元)**

- **writing-skills** - 根据最佳实践（包括测试方法）培养新技能
- **using-superpowers** - 使用超能力，技能系统介绍

### 哲学

- **Test-Driven Development** -（测试驱动开发）始终先写测试
- **Systematic over ad-hoc** -（系统化优于临时性）流程优于猜测
- **Complexity reduction** - （降低复杂性）以简洁为首要目标
- **Evidence over claims** -（以证据为准）在宣布成功之前先进行验证

## 5. What's Inside

### Skills Library

**Testing**

- **test-driven-development** - RED-GREEN-REFACTOR cycle (includes testing anti-patterns reference)

**Debugging**

- **systematic-debugging** - 4-phase root cause process (includes root-cause-tracing, defense-in-depth, condition-based-waiting techniques)
- **verification-before-completion** - Ensure it's actually fixed

**Collaboration**

- **brainstorming** - Socratic design refinement
- **writing-plans** - Detailed implementation plans
- **executing-plans** - Batch execution with checkpoints
- **dispatching-parallel-agents** - Concurrent subagent workflows
- **requesting-code-review** - Pre-review checklist
- **receiving-code-review** - Responding to feedback
- **using-git-worktrees** - Parallel development branches
- **finishing-a-development-branch** - Merge/PR decision workflow
- **subagent-driven-development** - Fast iteration with two-stage review (spec compliance, then code quality)

**Meta**

- **writing-skills** - Create new skills following best practices (includes testing methodology)
- **using-superpowers** - Introduction to the skills system

### Philosophy

- **Test-Driven Development** - Write tests first, always
- **Systematic over ad-hoc** - Process over guessing
- **Complexity reduction** - Simplicity as primary goal
- **Evidence over claims** - Verify before declaring success
