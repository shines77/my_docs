# git-如何获取远程分支

## 1. 获取远程分支

```bash
$ git pull -v --progress "origin"
$ git push --progress "origin" master:master

$ git config --global user.email "gz_shines@msn.com"
$ git config --global user.name "XiongHui Guo"
```

`git` 如何获取远程分支:

运行 `git fetch`，可以将远程分支信息获取到本地，再运行 `git checkout -b local-branchname origin/remote_branchname` 就可以将远程分支映射到本地命名为 `local-branchname` 的一分支。

```bash
$ git fetch --tags -v --progress    # 拉取所有远程分支的tag标签(tags包括分支和tag)信息
$ git fetch --all -v --progress     # 拉取所有远程分支(仅分支)的所有内容

$ git fetch --tags -v --progress    # --tags 只拉取远程分支的分支信息(仅分支列表), --all 拉取所有远程分支的所有数据

$ git fetch --tags --all -v --progress
$ git checkout -b jewel --track origin/jewel

或者

$ git checkout -b jewel origin/jewel

或者

$ git checkout --track origin/jewel

或者

$ git fetch --tags --all -v --progress
$ git branch jewel origin/jewel

$ git status 查看目前状态

$ git checkout -B stable/juno                           # 在本地创建一个叫stable/juno的新分支
$ git checkout -b stable/juno --track juno-eol          # 拉取远程分支juno-eol, 并且把分支重命名为stable/juno
```

切换分支:

```bash
$ git checkout 分支名
```

例如:

```bash
git fetch --tags --all -v --progress
git checkout -b next remotes/origin/next
git checkout next

或者

git fetch --tags --all -v --progress
git branch next remotes/origin/next
git checkout next
```

```bash
列出所有分支

$ git branch -a

列出所有 Tags

$ git tag -l

从某个 tag 复制并新建一个分支 branch

$ git checkout -b branch_name tag_name

删除分支

$ git branch -d stable/juno

删除一个远程分支

$ git push origin --delete remote/origin/master
```
