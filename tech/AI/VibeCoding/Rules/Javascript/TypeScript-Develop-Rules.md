# TypeScript开发规则

## 类型定义规则

1. 所有公共接口必须有明确的类型定义
2. 避免使用 `any` 类型，必要时使用 `unknown`
3. 使用严格的TSConfig配置
4. 自定义类型放在 `types/` 目录下

## 组件设计规则

1. React组件使用函数式写法
2. Props接口命名为 `{ComponentName}Props`
3. 使用 `forwardRef` 处理ref传递
4. 组件文件结构：导入 → 类型 → 组件 → 导出

## 状态管理规则

1. 本地状态优先使用 `useState`
2. 复杂状态使用 `useReducer`
3. 全局状态通过Context或状态管理库
4. 状态更新必须是不可变的

请在编写代码时严格遵循这些规则。
