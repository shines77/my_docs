# C++17 编程规范

## 基础规范

- **标准**: `C++17`，遵循 ISO 标准与现代 `C++` 最佳实践
- **文件结构**: 头文件 (*.h) + 实现文件 (*.cpp)
- **风格**: Google C++ Style Guide，缩进 2 空格，花括号不换行（main.cpp 除外）

## 命名约定

| 类型 | 规范 | 示例 |
|:---|:---|:---|
| 类/结构体 | PascalCase | `HttpRequest` |
| 变量/函数 | camelCase | `isUserSignedIn` |
| 全局常量/宏 | SCREAMING_SNAKE_CASE | `MAX_BUFFER_SIZE` |
| 类常量 | kPascalCase | `kDefaultTimeout` |
| 类成员变量 | 下划线前缀 | `_userId` |
| 结构体成员 | camelCase，无前缀 | `userId` |

## 核心原则

- **指针**: 默认裸指针，必要时用智能指针
- **内存**: 优先栈分配，用 RAII 管理资源
- **字符串**: 用 `std::string_view` / `llvm::StringRef` 避免拷贝
- **现代特性**: `auto`、范围 for、`constexpr`、`std::optional`/`variant`
- **异常**: 用标准异常处理错误，函数边界验证输入
- **性能**: 移动语义 + `noexcept`，用 `<algorithm>` 替代手写循环

## 禁止事项

- 避免全局变量，谨慎使用单例
- 不用 `enum class`（除非必要）
- 不用 C 风格类型转换
- 不用原始数组，用 `std::array`/`std::vector`

## 其他

- **注释**: 用英文，详尽且清晰，Token代表的具体符号添加注释，方便阅读，API 文档用 `///`
- **测试**: 单元测试 + 集成测试
- **日志**: 使用 spdlog 等日志库