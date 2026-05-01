# OpenFang — Agent Instructions

## 项目概览

OpenFang 是一个使用 Rust 编写的开源智能体操作系统（包含 14 个 crate）。

- Config: `~/.openfang/config.toml`
- Default API: `http://127.0.0.1:4200`
- CLI binary: `target/release/openfang.exe` (or `target/debug/openfang.exe`)

## 构建与验证工作流

在实现每一项功能之后，请运行全部三项检查：

```bash
cargo build --workspace --lib          # Must compile (use --lib if exe is locked)
cargo test --workspace                 # All tests must pass (currently 1744+)
cargo clippy --workspace --all-targets -- -D warnings  # Zero warnings
```

## 强制要求：实时集成测试

**在实现任何新端点、功能或进行内部连接变更之后，您必须运行实时集成测试。** 仅靠单元测试是不够的——即使单元测试通过了，该功能实际上可能仍是“死代码”（即从未被执行的代码）。实时测试能够捕获以下问题：

- `server.rs` 文件中遗漏了路由注册
- 配置文件（TOML）中的配置字段未能正确反序列化
- 内核层与 API 层之间存在类型不匹配
- 能够编译通过，但实际返回错误数据或空数据的端点（Endpoints）

### 如何运行实时集成测试

#### 步骤 1：停止所有正在运行的守护进程 (daemon)

```bash
tasklist | grep -i openfang
taskkill //PID <pid> //F
# Wait 2-3 seconds for port to release
sleep 3
```

#### 步骤 2：构建新的发布二进制文件

```bash
cargo build --release -p openfang-cli
```

#### 步骤 3：使用所需的 API 密钥启动守护进程 (daemon)

```bash
GROQ_API_KEY=<key> target/release/openfang.exe start &
sleep 6  # Wait for full boot
curl -s http://127.0.0.1:4200/api/health  # Verify it's up
```

守护进程命令是 `start`（而不是 `daemon`）。

#### 步骤 4：测试每一个新端点 (endpoint)

```bash
# GET endpoints — verify they return real data, not empty/null
curl -s http://127.0.0.1:4200/api/<new-endpoint>

# POST/PUT endpoints — send real payloads
curl -s -X POST http://127.0.0.1:4200/api/<endpoint> \
  -H "Content-Type: application/json" \
  -d '{"field": "value"}'

# Verify write endpoints persist — read back after writing
curl -s -X PUT http://127.0.0.1:4200/api/<endpoint> -d '...'
curl -s http://127.0.0.1:4200/api/<endpoint>  # Should reflect the update
```

#### 步骤 5：测试实际的 LLM 集成

```bash
# Get an agent ID
curl -s http://127.0.0.1:4200/api/agents | python3 -c "import sys,json; print(json.load(sys.stdin)[0]['id'])"

# Send a real message (triggers actual LLM call to Groq/OpenAI)
curl -s -X POST "http://127.0.0.1:4200/api/agents/<id>/message" \
  -H "Content-Type: application/json" \
  -d '{"message": "Say hello in 5 words."}'
```

#### 步骤 6：验证副作用

After an LLM call, verify that any metering/cost/usage tracking updated:
```bash
curl -s http://127.0.0.1:4200/api/budget       # Cost should have increased
curl -s http://127.0.0.1:4200/api/budget/agents  # Per-agent spend should show
```

#### 步骤 7：验证仪表板(dashboard) HTML

```bash
# Check that new UI components exist in the served HTML
curl -s http://127.0.0.1:4200/ | grep -c "newComponentName"
# Should return > 0
```

#### 步骤 8: 清理

```bash
tasklist | grep -i openfang
taskkill //PID <pid> //F
```

### 测试 Key API 端点 (endpoint)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/health` | GET | Basic health check |
| `/api/agents` | GET | List all agents |
| `/api/agents/{id}/message` | POST | Send message (triggers LLM) |
| `/api/budget` | GET/PUT | Global budget status/update |
| `/api/budget/agents` | GET | Per-agent cost ranking |
| `/api/budget/agents/{id}` | GET | Single agent budget detail |
| `/api/network/status` | GET | OFP network status |
| `/api/peers` | GET | Connected OFP peers |
| `/api/a2a/agents` | GET | External A2A agents |
| `/api/a2a/discover` | POST | Discover A2A agent at URL |
| `/api/a2a/send` | POST | Send task to external A2A agent |
| `/api/a2a/tasks/{id}/status` | GET | Check external A2A task status |

## 架构笔记

- **请勿改动 `openfang-cli`** —— 该交互式 CLI 目前正由用户积极开发中
- `KernelHandle` Trait 用于避免运行时（Runtime）与内核（Kernel）之间产生循环依赖
- `server.rs` 中的 `AppState` 结构体充当桥梁，连接内核与 API 路由
- 新增路由必须**既**在 `server.rs` 的路由器中进行注册，**又**在 `routes.rs` 中实现其具体逻辑
- 仪表板（Dashboard）是一个基于 Alpine.js 构建的单页应用（SPA），位于 `static/index_body.html` 文件中 —— 新增的标签页（Tabs）需要同时包含相应的 HTML 结构以及 JS 数据与方法
- 配置字段（Config fields）的定义需要包含以下要素：结构体字段声明 + `#[serde(default)]` 属性 + `Default` Trait 的实现入口 + `Serialize` 与 `Deserialize` 的派生宏

## 常见陷阱

- 如果守护进程 (daemon) 正在运行，`openfang.exe` 文件可能会被锁定——请使用 `--lib` 标志，或者先终止守护进程
- `PeerRegistry` 在内核（Kernel）中是 `Option<PeerRegistry>` 类型，但在 `AppState` 中却是 `Option<Arc<PeerRegistry>>` 类型——请使用 `.as_ref().map(|r| Arc::new(r.clone()))` 进行封装转换
- 添加到 `KernelConfig` 结构体中的配置字段，**必须**同时也添加到其 `Default` 实现中，否则会导致构建失败
- `AgentLoopResult` 结构体中的响应字段是 `.response`，而不是 `.response_text`
- 用于启动守护进程的 CLI 命令是 `start`，而不是 `daemon`
- 在 Windows 系统上：请使用 `taskkill //PID <pid> //F` 命令（若在 MSYS2 或 Git 的 Bash 环境下执行，请使用双斜杠）
