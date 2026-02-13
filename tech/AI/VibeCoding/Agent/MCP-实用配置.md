# MCP 服务器的实用配置

## MPC 配置

参考仓库中的配置模式，一个高效的 MCP 配置应该包括：

开发环境 MCP 配置：

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "mcp-server-filesystem",
      "args": ["./src", "./docs", "./tests"]
    },
    "git": {
      "command": "mcp-server-git",
      "args": ["--repo-path", "."]
    },
    "database": {
      "command": "mcp-server-postgres",
      "args": ["--connection", "postgresql://localhost:5432/mydb"]
    },
    "api-testing": {
      "command": "mcp-server-http",
      "args": ["--base-url", "http://localhost:3000/api"]
    }
  }
}
```

**对应的使用技巧：**

请通过MCP检查当前项目的文件结构，然后：

1. 分析现有的 API 路由
2. 检查数据库表结构
3. 查看最近的 Git 提交记录
4. 基于这些信息为新功能制定实施计划
