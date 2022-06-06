# Git 忽略不提交文件的 3 种情形

## 1. 从未提交过的文件

`.gitignore` 文件 ：从未提交过的文件，从来没有被 `git` 记录过的文件。

也就是添加之后从来没有提交（commit）过的文件，可以使用 `.gitignore` 忽略该文件，只能作用于未跟踪的文件（Untracked Files）。

例如：

`vim .gitignore`

```text
.DS_Store
node_modules/
dist/
npm-debug.log
build-debugger.js
.idea
```

支持通配符 `*` 和 `**`，例如：

```text
# 只忽略 /upload/ 目录下的 *.h 文件，而不能忽略 /upload/001/ 目录下的 `*.h` 文件
/upload/*.h

# 忽略 /projects/ 目录下的所有 *.h 文件（包括子目录中的）
/projects/**/*.h
```

## 2. 已经提交过的文件

命令：

```shell
git rm --cached Xml/config.xml
```

已经推送（push）过的文件，想从 `git` 远程库中删除，并在以后的提交中忽略，但是却还想在本地保留这个文件，执行该命令。`Xml/config.xml` 是要从远程库中删除的文件或目录，支持通配符 `*` 。

比如，不小心提交到 `git` 上的一些 `log` 日志文件，想从远程库删除，可以用这个命令。

这个命令在提交后，并推送到远程 `git` 端时，把远程仓库的指定文件删除。

先提交本地删除（不会删除文件本身），再把该文件或目录写入 `.gitignore` 中，即可让 `.gitignore` 生效。

## 3. 已提交过但不希望被修改的文件

如果想让某个文件后续的更改都不会更新到远端的 `git` 仓库上，可以使用该命令。

命令：

```shell
git update-index --assume-unchanged Xml/config.xml
```

已经推送（push）过的文件，想在以后的提交时忽略此文件，即使本地已经修改过，而且也不会删除 `git` 远程库中相应的文件。`Xml/config.xml` 是不想更新的文件或目录。

适用于：`git` 远程库上有一个标准配置文件，然后每个人根据自己的具体配置，修改一份自用，但又不会将该配置文件提交到远程库中。

如果想取消不更新，则执行：

```shell
git update-index --no-assume-unchanged Xml/config.xml
```

## 4. 参考文章

1. `[git 忽略不提交的文件 3 种情形]`

    [https://www.cnblogs.com/alice-fee/p/6757301.html](https://www.cnblogs.com/alice-fee/p/6757301.html)

2. `[git使用-忽略文件更新的几种方法]`

    [https://www.cnblogs.com/sutrahsing/p/11752092.html](https://www.cnblogs.com/sutrahsing/p/11752092.html)

3. `[git文件锁定不更新和忽略]`

    [https://www.cnblogs.com/cu-later/p/13802011.html](https://www.cnblogs.com/cu-later/p/13802011.html)
