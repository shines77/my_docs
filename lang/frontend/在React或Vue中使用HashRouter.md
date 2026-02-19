# 在 React 或 Vue 中使用 HashRouter

## 1. React

### 1.1 首先安装 React Router

```bash
npm install react-router-dom
# 或
yarn add react-router-dom
```

如果依赖有冲突，则使用：

```bash
npm install react-router-dom --legacy-peer-deps
```

检查是否安装成功，以及版本号：

```bash
npm list react-router-dom 2>&1 | head -5

# 结果中显示如下字样：
-- react-router-dom@7.13.0
```

### 1.2 创建路由配置文件

在 src/router/index.jsx 或 src/router/index.js 中：

```javascript
// src/router/index.jsx 或 src/router/index.js
import { createHashRouter, RouterProvider } from 'react-router-dom'
import Home from '../views/Home'
import About from '../views/About'

// 使用 createHashRouter 创建路由
const router = createHashRouter([
  {
    path: '/',
    element: <Home />,
  },
  {
    path: '/about',
    element: <About />,
  }
])

export default router
```

如果你使用了 vite 来构建，可能需要安装 terser，命令如下：

```bash
npm install -D terser --legacy-peer-deps
```
