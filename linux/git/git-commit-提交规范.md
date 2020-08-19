
# git commit 提交规范

## 1. 提交格式

`git` 提交格式:

```
<type>(<scope>): <subject>
<BLANK LINE>
<body>
<BLANK LINE>
<footer>
```

大致分为三个部分(使用空行分割):

* 标题行: 必填, 描述主要修改类型和内容
* 主题内容: 描述为什么修改, 做了什么样的修改, 以及开发的思路等等
* 页脚注释: 放 Breaking Changes 或 Closed Issues

* type: commit 的类型

```
init: 初始化
update: 更新一些文件的内容
modified: 微小的修改
style: 代码格式修改
bugfix: 修复bug
feature: 新功能
refactor: 代码重构
optimize: 性能优化
pref: 优化相关, 比如提升性能, 体验
docs: 文档修改
test: 测试用例修改
build: 构建项目
revert: 回滚到上一个版本
chore: 其他修改, 比如依赖管理。chore 的中文翻译为日常事务、例行工作，顾名思义，即不在其他 commit 类型中的修改，都可以用 chore 表示。
```

* scope: commit 影响的范围, 比如: route, component, utils, build...

* subject: commit 的概述

* body: commit 具体修改内容, 可以分为多行.

* footer: 一些备注, 通常是 BREAKING CHANGE 或修复的 bug 的链接.

## 2. 参考文章:

* [git commit 提交规范](https://zhuanlan.zhihu.com/p/90281637)

* [Git Commit 标准化](https://www.cnblogs.com/wubaiqing/p/10307605.html)
