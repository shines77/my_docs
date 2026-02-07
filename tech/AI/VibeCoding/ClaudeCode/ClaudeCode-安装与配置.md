# ClaudeCode 安装与配置

Anthropic 官网：[https://www.anthropic.com/](https://www.anthropic.com/)
Claude 官网：[https://claude.ai/](https://claude.ai/)

## 1. 安装 Claude Code

### 1.1 命令行安装

macOS, Linux, WSL:

```bash
curl -fsSL https://claude.ai/install.sh | bash
```

Windows PowerShell:

```bash
irm https://claude.ai/install.ps1 | iex
```

Windows CMD:

```bash
curl -fsSL https://claude.ai/install.cmd -o install.cmd && install.cmd && del install.cmd
```

（注：Windows 上需要以管理员身份运行 PowerShell 或 CMD 。）

安装完成后，验证一下:

```bash
claude --version
```

如果显示版本号，说明安装成功！

> Native installations automatically update in the background to keep you on the latest version.

### 1.2 npm 安装

你的电脑需要安装了 Node.js，在命令行输入：

```bash
node --version
```

如果显示版本号(比如 v18.17.0)，说明已安装。如果没有，去 nodejs.org 下载安装。

（注：Windows 上需要以管理员身份运行 PowerShell 或 CMD 。）

打开命令行，输入以下命令:

```bash
npm install -g @anthropic-ai/claude-code
```

等待安装完成(可能需要几分钟)。

安装完成后，验证一下:

```bash
claude --version
```

如果显示版本号，说明安装成功！

## 2. 登录 Claude Code

### 首次登录流程

当你第一次启动 Claude Code 时，需要登录账号：

```bash
claude
```

注：由于 anthropic 拒绝中国 IP 登录，所以这里需要科学上网，而且注册的时候需要收手机验证码，但又不支持国内的手机号。这里，可以用 Google 账户来登录，这样可以绕开手机验证码的问题。

启动后，系统会提示你登录，在 Claude Code 界面中输入：

```bash
/login
```

然后按照提示完成登录，先选择免费的套餐。订阅的话，也不支持国内的信用卡，比较麻烦。

如果想绕开 anthropic 的登录，不用科学上网也等使用，可以添加如下设置。

### 编辑配置文件

编辑或新增 `.claude.json` 文件，MacOS & Linux 为 `~/.claude.json`，Windows 为用户目录 `%USERPROFILE%\.claude.json` 。

```bash
# 在 json 文件头部或尾部，新增 `hasCompletedOnboarding` 参数
{
  "hasCompletedOnboarding": true,

  ..............
  ... 原来的 json 内容 ...
  ..............
}
```

然后再次启动 Claude 时，则不用再登录 anthropic 账号。

## 3. 启动第一次会话

### 在项目目录中启动

打开终端，进入你的项目目录，然后启动 Claude Code：

```bash
cd /path/to/your/project
claude
```

### 信任文件夹

启动后，选择 信任此文件夹 (Trust This Folder)，以允许 Claude Code 访问该文件夹中的文件 。

### 一些命令

```bash
/help      帮助菜单
/clear     清除会话历史并释放上下文
/compact   清除会话历史，但保留上下文的概要
/resume    恢复之前的对话
/agents    管理 agent 设置
/model     当前选择的模型，已经模型列表

/config    显示设置页
/context   用一个彩色的 Grid 可视化的显示当前的上下文
/cost      显示当前会话的 token 消耗
/doctor    诊断和验证 Claude Code 的安装和设置
/exit      退出
```

## 4. 基本命令

```bash
# 更新 Claude Code
claude update

# 启动 MCP 向导
claude mcp

# 指定项目路径，并启动
claude --project /path/to/project

# 查看项目状态
claude /project-info

# 重置项目设置
claude /reset-project

# 创建 Git 提交
claude commit

# 运行一次性任务
claude "task"

# 运行一次性查询，然后退出
claude -p "query"

# 在当前目录中继续最近的对话
claude -c

# 恢复之前的对话
claude -r
```

## 5. 配置系统

### 5.1 全局配置

```bash
# 查看所有配置
claude config list

# 设置全局配置
claude config set --global model claude-opus-3
claude config set --global max-tokens 4000
claude config set --global temperature 0.7

# 重置所有配置
claude config reset
```

### 5.2 项目配置

```bash
# 在项目目录中设置
claude config set --project model claude-sonnet-3.5
claude config set --project ignore-patterns "*.log,temp/*"

# 查看项目配置
claude config list --project

# 继承全局配置
claude config inherit --global
```

### 5.3 常用配置命令

```bash
# 设置字符串值
claude config set model "claude-3-sonnet"

# 设置数值
claude config set maxTokens 4000

# 设置布尔值
claude config set verbose true

# 设置数组
claude config set ignorePatterns "*.log" "temp/*" "node_modules"
```

### 5.4 查看配置

```bash
# 查看所有配置
claude config list

# 查看特定配置
claude config get model

# 查看配置源
claude config source model
```

### 5.5 管理配置

```bash
# 删除配置
claude config unset verbose

# 重置所有配置
claude config reset

# 导出配置
claude config export > config-backup.json

# 导入配置
claude config import < config-backup.json
```

## 6. MCP 集成 ​

MCP (Model Context Protocol) 允许 Claude Code 连接外部服务和数据库。

### 6.1 启用 MCP ​

```bash
# 启动 MCP 配置向导
claude mcp

# 手动配置 MCP
claude config set mcp.enabled true
```

### 6.2 常用 MCP 服务器

数据库连接 ​

```bash
# PostgreSQL
claude mcp add postgresql --connection-string "postgresql://user:pass@localhost:5432/db"

# MySQL
claude mcp add mysql --host localhost --user root --database mydb

# SQLite
claude mcp add sqlite --database-path ./data.db
```

文件系统

```bash
# 本地文件系统
claude mcp add filesystem --root-path /path/to/project

# Git 仓库
claude mcp add git --repository-path /path/to/repo
```

Web 服务

```bash
# REST API
claude mcp add rest-api --base-url https://api.example.com --auth-token your-token

# GraphQL
claude mcp add graphql --endpoint https://api.example.com/graphql
```

## 7. 安全和权限管理

### 7.1 工具权限控制

```bash
# 查看可用工具
claude /tools

# 限制工具权限
claude config set allowedTools "Edit,View,Terminal"

# 禁用危险工具
claude config set deniedTools "Delete,Execute"

# 设置工具白名单
claude config set toolWhitelist "git,npm,pip,cargo"
```

### 7.2 文件访问控制

```bash
# 设置忽略模式
claude config set ignorePatterns ".env,.secrets,*.key,id_rsa*"

# 设置只读目录
claude config set readOnlyPaths "/etc,/var,/usr"

# 设置禁止访问的目录
claude config set forbiddenPaths "/root,/home/*/private"
```

## 8. 思考模式

Claude Code 支持深度思考模式，适用于复杂问题。

### 8.1 启用思考模式

```bash
# 在交互模式中
/think 如何优化这个算法的时间复杂度？

# 命令行模式
claude --think "设计一个分布式缓存系统"

# 深度思考
claude --deep-think "分析这个系统的安全漏洞"
```

### 8.2 思考模式配置

```bash
# 设置思考超时时间
claude config set thinkTimeout 300

# 启用思考记录
claude config set saveThoughts true

# 查看思考历史
claude /thoughts
```


## 9. 团队协作与自动化

### 9.1 Git 集成

```bash
# 自动审查代码
claude "请审查我的最新提交"

# 生成提交信息
git diff --cached | claude -p "生成简洁的提交信息"

# 代码重构建议
claude "分析这个分支的代码质量并提出改进建议"
```

### 9.2 CI/CD 集成

#### GitHub Actions 示例

```bash
name: Claude Code Review
on: [pull_request]
jobs:
 review:
 runs-on: ubuntu-latest
 steps:
 - uses: actions/checkout@v2
 - name: Install Claude Code
 run: npm install -g @anthropic-ai/claude-code
 - name: Code Review
 run: |
 git diff origin/main...HEAD | claude -p "审查这些代码更改" > review.md
 gh pr comment --body-file review.md
```

#### Jenkins Pipeline 示例

```bash
pipeline {
 agent any
 stages {
 stage('Code Analysis') {
 steps {
 script {
 sh 'claude "分析项目代码质量" > analysis.txt'
 archiveArtifacts 'analysis.txt'
 }
 }
 }
 }
}
```

### 9.3 自动化脚本

#### 代码质量检查脚本

```bash
#!/bin/bash
# code-quality-check.sh

echo "🔍 正在进行代码质量检查..."

# 检查代码风格
claude "检查代码风格是否符合团队规范" > style-check.txt

# 检查安全漏洞
claude "扫描代码中的潜在安全漏洞" > security-check.txt

# 性能分析
claude "分析代码性能瓶颈" > performance-check.txt

echo "✅ 代码质量检查完成，报告已生成"
```

#### 文档生成脚本

```bash
#!/bin/bash
# generate-docs.sh

echo "📝 正在生成文档..."

# 生成 API 文档
claude "为这个项目生成详细的 API 文档" > api-docs.md

# 生成使用指南
claude "生成用户使用指南" > user-guide.md

# 生成开发者文档
claude "生成开发者文档和部署指南" > dev-docs.md

echo "✅ 文档生成完成"
```

## 10. 高级特性

### 10.1 插件系统

```bash
# 查看已安装插件
claude plugin list

# 安装插件
claude plugin install claude-eslint
claude plugin install claude-pytest

# 启用/禁用插件
claude plugin enable claude-eslint
claude plugin disable claude-pytest

# 更新插件
claude plugin update
```

### 10.2 模板系统 ​

```bash
# 创建代码模板
claude template create --name "react-component" --path ./templates/

# 使用模板
claude template use react-component --name MyComponent

# 列出模板
claude template list

# 分享模板
claude template share react-component
```

### 10.3 工作空间管理

```bash
# 创建工作空间
claude workspace create my-project

# 切换工作空间
claude workspace switch my-project

# 列出工作空间
claude workspace list

# 删除工作空间
claude workspace delete my-project
```

### 10.4 性能优化

```bash
# 启用缓存
claude config set cache.enabled true
claude config set cache.ttl 3600

# 设置并发限制
claude config set maxConcurrent 5

# 启用增量处理
claude config set incrementalMode true

# 性能监控
claude /performance
```

## 11. 故障排除

### 常见问题解决方案

#### 1. 安装问题

```bash
# 权限问题
sudo npm install -g @anthropic-ai/claude-code

# 网络问题
npm config set registry https://registry.npmmirror.com
npm install -g @anthropic-ai/claude-code

# 版本冲突
npm uninstall -g @anthropic-ai/claude-code
npm cache clean --force
npm install -g @anthropic-ai/claude-code@latest
```

#### 2. 配置问题

```bash
# 重置所有配置
claude config reset

# 检查环境变量
echo $ANTHROPIC_API_KEY

# 诊断和验证安装
claude /doctor
```

#### 3. 连接问题

```bash
# 测试连接
claude "test connection"

# 检查代理设置
claude config get proxy

# 更换 API 端点
claude config set apiEndpoint "https://api.anthropic.com"
```

## 12. 一些设置

**打开 agent team（蜂群）模式。**

配置文件加入下列的设置：

macOS路径：`~/.claude.json`

Windows 路径：`%USERPROFILE%\.claude.json`

```json
{
  ............
  ..其他设置..
  ............ ,

  "env": {
    "CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS": "1"
  }
}
```

## 13. 交互技巧

### 在 Claude Code 中使用 Git

Claude Code 使 Git 操作变得对话式：

```text
> what files have I changed?
```

```text
> commit my changes with a descriptive message
```

您也可以提示更复杂的 Git 操作：

```text
> create a new branch called feature/quickstart
```

```text
> show me the last 5 commits
```

```text
> help me resolve merge conflicts
```

### 修复错误或添加功能

Claude 擅长调试和功能实现。

用自然语言描述您想要的内容：

```text
> add input validation to the user registration form
```

或修复现有问题：

```text
> there's a bug where users can submit empty forms - fix it
```

Claude Code 将：

- 定位相关代码
- 理解上下文
- 实现解决方案
- 如果可用，运行测试

### 其他常见工作流

有许多方式可以与 Claude 合作：

**重构代码**

```text
> refactor the authentication module to use async/await instead of callbacks
```

**编写测试**

```text
> write unit tests for the calculator functions
```

**更新文档**

```text
> update the README with installation instructions
```

**代码审查**

```text
> review my changes and suggest improvements
```

### 使用建议

**Claude Code 是你的 AI 结对编程伙伴**

- 像跟有帮助的同事交流一样跟它对话
- 描述你想实现什么,它会帮你达成目标
- 不用拘泥于命令格式,用自然语言表达即可

## x. 总结

Claude Code 作为革命性的 CLI 编程助手，提供了：

1. **强大的 AI 能力** - 原生 Claude 4.5 Sonnet/Opus 支持
2. **完整的工具生态** - MCP 集成、插件系统、模板管理
3. **企业级安全** - 权限控制、访问限制、审计日志
4. **团队协作** - Git 集成、CI/CD 支持、自动化脚本
5. **高度可配置** - 灵活的配置系统、工作空间管理

通过本指南的学习，你将能够充分发挥 Claude Code 的强大功能，提升开发效率和代码质量。

## y. 参考文章

- [Claude Code 安装与使用](https://www.runoob.com/claude-code/claude-code-install.html)
- [Claude Code 完整使用指南](https://www.claude-cn.org/posts/Claude-code-complete-guide.html)
- [官网：Claude Code 快速入门](https://code.claude.com/docs/zh-CN/quickstart)
