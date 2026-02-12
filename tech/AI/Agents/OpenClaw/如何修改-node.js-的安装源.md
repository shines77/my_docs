
# 如何修改 node.js 的安装源

以下是几种常用的更改 npm 源的方法：

## 方法1：手动设置

您可以直接运行以下命令之一：

```bash
# 设置为淘宝镜像源（推荐国内用户）
npm config set registry https://registry.npmmirror.com/

# 设置为腾讯云镜像源
npm config set registry https://mirrors.cloud.tencent.com/npm/

# 设置为华为云镜像源
npm config set registry https://repo.huaweicloud.com/repository/npm/

# 恢复官方源
npm config set registry https://registry.npmjs.org/
```

## 方法2：临时使用

如果只想在单次命令中使用特定源：

```bash
# 临时使用淘宝源安装包
npm install --registry https://registry.npmmirror.com/ [package-name]
```

## 方法3：使用 nrm 管理源

安装 nrm（npm registry manager）工具：

```bash
# 安装 nrm
npm install -g nrm

# 查看可用源
nrm ls

# 切换到淘宝源
nrm use taobao

# 测试源速度
nrm test
```
