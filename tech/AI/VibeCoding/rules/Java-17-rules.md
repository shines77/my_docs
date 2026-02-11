# Java 开发规范/编码规则

## 概述

包含了从 Java 版本、命名规范、文档注释、不可变性、Lombok 使用、异常处理、日志记录、Spring 依赖注入到单元测试的完整开发规范。

- 我使用 Java 17 或更高版本
- 遵循标准的 Java 命名规范：类名用 PascalCase，方法和变量用 camelCase，常量全大写加下划线
- 所有 public 方法必须包含完整的 Javadoc 注释，包括 @param、@return 和 @throws
- 优先使用不可变对象和 final 关键字（如 final class、final fields）
- 避免使用 Lombok；显式编写 getter/setter 和构造函数
- 异常处理应具体，避免捕获通用 Exception，优先抛出或记录有意义的异常
- 使用 SLF4J 进行日志记录，而不是 System.out
- 依赖注入使用 Spring Framework（如适用），偏好构造器注入而非字段注入
- 单元测试使用 JUnit 5 + Mockito，每个业务类都应有对应的测试类
