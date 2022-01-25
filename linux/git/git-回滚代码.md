# git 回滚代码版本的几种方式

## 1. 回滚代码

`Git` 回滚代码到某个 `commit`

回退命令：

```shell
git reset --hard HEAD^      回退到上个版本

git reset --hard HEAD~3     回退到前 3 次提交之前，以此类推，回退到 n 次提交之前

git reset --hard commit_id  退到/进到，指定 commit 的哈希码（这次提交之前或之后的提交都会回滚）

例如:

git reset --hard 7e19f90a75b4a266b78523adaefb0d3ddff11c69
```

## 2. 参考文章

原文链接：[git 回滚代码版本的几种方式](https://blog.csdn.net/xinzhifu1/article/details/92770108)
