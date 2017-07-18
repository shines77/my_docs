[GIT] 如何同步一个 fork 后的同步源的新更新内容
===============================================

关键字：`git`，`gitlab`，`github.com`，`fork`，`fork sync`，`Syncing a fork`。

# 1. 概要 #

`GitHub` 或 `GitLab` 上 `fork` 了别人的仓库后，原作者又更新了仓库，如何将自己的代码和原仓库保持一致？本文将为你解答。

# 2. GitHub 官网文档 #

[Configuring a remote for a fork](https://link.zhihu.com/?target=https%3A//help.github.com/articles/configuring-a-remote-for-a-fork/)

[Syncing a fork](https://link.zhihu.com/?target=https%3A//help.github.com/articles/syncing-a-fork/)

以上两个链接是 `GitHub` 官方的帮助文档中关于 `fork同步` 的说明，中文的链接请看文末的 `X. 参考文章` 小节。

# 3. 具体方法 #

## 3.1. 为 fork 配置远端库 ##

### 3.1.1. 查看远端状态 ###

使用下列的命令，查看当前仓库的远端状态，例如：

```shell
$ git remote -v

origin	https://github.com/hwf452/my_docs.git (fetch)
origin	https://github.com/hwf452/my_docs.git (push)
```

注：这里的 `https://github.com/hwf452/my_docs.git` 就是你从别人那 `fork` 来的仓库。

默认仓库的远端状态格式是：

```shell
origin  https://github.com/YOUR_USERNAME/YOUR_FORK.git (fetch)
origin  https://github.com/YOUR_USERNAME/YOUR_FORK.git (push)
```

### 3.1.2. 添加原始远端仓库 ###

添加一个远端仓库 `upstream`（即我们要同步的原始仓库），例如：

```shell
$ git remote add upstream https://github.com/shines77/my_docs.git
```

即 `https://github.com/hwf452/my_docs.git` 仓库是从 `https://github.com/shines77/my_docs.git` 仓库 `fork` 而来的，用户 `shines77` 是 `my_docs` 仓库的原始作者，。

添加原始远端仓库的格式一般是：

```shell
$ git remote add upstream https://github.com/ORIGINAL_OWNER/ORIGINAL_REPOSITORY.git
```

其中 `upstream` 叫 `original_repo`，`sync_repo` 都是可以的，怎么好记怎么用。

### 3.1.3. 再次检查远端状态 ###

再次查看远端状态，以确认是否配置成功。

```shell
$ git remote -v

origin    https://github.com/hwf452/my_docs.git (fetch)
origin    https://github.com/hwf452/my_docs.git (push)
upstream  https://github.com/shines77/my_docs.git (fetch)
upstream  https://github.com/shines77/my_docs.git (push)
```

## 3.2. 同步一个 fork ##

### 3.2.1. 获取远端仓库 ###

从原始远端仓库 `fetch` 分支和提交点，提交给本地 `master`，并会被存储在一个本地分支 `upstream/master` 里：

```shell
$ git fetch upstream

remote: Counting objects: 75, done.
remote: Compressing objects: 100% (53/53), done.
remote: Total 62 (delta 27), reused 44 (delta 9)
Unpacking objects: 100% (62/62), done.
From https://github.com/shines77/my_docs.git
 * [new branch]      master     -> upstream/master
```

### 3.2.2. 切换本地分支 ###

切换到本地主分支（如果当前分支不是 `master` 分支的话，当前分支状态可以使用 “`git status`” 命令查看。）

```shell
$ git checkout master

Already on 'master'
Your branch is up-to-date with 'origin/master'.
```

### 3.2.3. 合并分支 ###

把 `upstream/master` 分支合并到本地的 `master` 分支上，这样就完成了同步，并且不会丢掉本地修改的内容。

```shell
$ git merge upstream/master

Updating a422352..5fdff0f
Fast-forward
 README                    |    9 -------
 README.md                 |    7 ++++++
 2 files changed, 7 insertions(+), 9 deletions(-)
 delete mode 100644 README
 create mode 100644 README.md
```

### 3.2.4. 推送到自己的仓库 ###

刚才已经完成从原始仓库的 `upstream/master` 到本地 `master` 的合并，现在把合并后的结果推送到自己的仓库，以完成更新：

```shell
$ git push origin master

(此处推送的结果省略......)
```

即把本地合并后的 `master` 分支推送到远端的 `origin/master` 分支上，即你自己的仓库，以此完成更新。

# 4. 其他 #

同理，我们也可以从自己的仓库，把修改后的代码提交到原始仓库上，但一般在原始仓库上，我们是没有提交权限的，所以还是改为提交 `PR` (`Pull Requests`) 比较好。

# 5. 参考文章 #

1. [\[知乎\] gitlab或github下fork后如何同步源的新更新内容？](https://www.zhihu.com/question/28676261)

2. [同步一个 fork （中文博客）](https://gaohaoyang.github.io/2015/04/12/Syncing-a-fork/)

3. [github.com 官方文档：Syncing a fork (英文)](https://help.github.com/articles/syncing-a-fork/)

<.End.>
