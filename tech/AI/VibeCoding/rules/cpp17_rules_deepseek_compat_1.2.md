# C++17 编程规范

## 核心原则

- **标准**：`C++17`，遵循`ISO`现代`C++`最佳实践
- **范式**：酌情使用面向对象、过程式、函数式
- **库**：优先STL和标准算法
- **结构**：分离头文件(.h)与实现文件(.cpp)

## 代码风格

- **基准**：Google风格 (Google C++ Style Guide)
- **命名空间**：顶格，内容不缩进
- **花括号**：同行（除`main.cpp`外）
- **缩进**：2空格
- **注释**：仅英文，详尽且清晰，Token代表的具体符号添加注释，方便阅读

## 命名规则

| 类别 | 格式 | 示例 |
|------|------|------|
| 类 | `PascalCase` | `UserAccount` |
| 变量/方法 | `camelCase` | `userName`, `getData()` |
| 全局常量/宏 | `SCREAMING_SNAKE_CASE` | `MAX_SIZE` |
| 类内常量 | `kPascalCase` | `kMaxRetry` |
| 成员变量（class） | `_camelCase` | `_userId` |
| 成员变量（struct） | `camelCase` | `userId` |
| bool函数 | `isClassOf()` | `isValid()` |
| 命名原则 | 描述性，少缩写（>9字符可酌情） | - |

## C++17特性

- ✅ **优先使用**：`auto`, 范围for, 智能指针, `string_view`, `optional`, `variant`, `constexpr`
- ⚠️ **内存管理**：**默认裸指针**；仅必要时使用智能指针，确保无性能/安全风险
- ❌ **避免**：`enum class`（除非必要）, C风格转换, 原始数组

## 错误处理

- 异常：`std::runtime_error`, `std::invalid_argument`等
- RAII管理资源
- 函数入口校验参数
- 日志：推荐spdlog

## 性能要点

- 栈优先，避免堆分配
- 移动语义 + `noexcept`
- 标准算法替代手写循环

## 安全规范

- 防缓冲区溢出、悬空指针
- 常量正确性（`const`成员、参数）
- 强制转换：`static_cast`等，禁用C风格
- 容器：`std::array`/`vector` > 原始数组

## 文档

- API注释：`///` 前缀
- 记录假设、约束、预期行为

## 测试

- 单元测试：单功能点
- 集成测试：组件级