# Git 忽略不提交文件的 3 种情形

## 1. 从未提交过的文件

`.gitignore` 文件 ：从未提交过的文件，从来没有被 Git 记录过的文件。

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
# 只忽略 /upload/ 目录下的 *.h 文件，而不能忽略 /upload/001/ 目录下的 *.h 文件
/upload/*.h

# 忽略 /projects/ 目录下的所有 *.h 文件（包括子目录中的）
/projects/**/*.h
```

## 2. 已经提交过的文件

命令：

```shell
git rm --cached Xml/config.xml
```

已经推送（push）过的文件，想从 git 远程库中删除，并在以后的提交中忽略，但是却还想在本地保留这个文件执行该命令。后面的 Xml/config.xml 是要从远程库中删除的文件的路径，支持通配符 `*` 。

比如，不小心提交到 git 上的一些 log 日志文件，想从远程库删除，可以用这个命令。

## 3. 已提交过但不希望被修改的文件

命令：

```shell
git update-index --assume-unchanged Xml/config.xml
```

已经推送（push）过的文件，想在以后的提交时忽略此文件，即使本地已经修改过，而且不删除 git 远程库中相应文件。后面的 Xml/config.xml 是要忽略的文件的路径。

适用于：git 远程库上有一个标准配置文件，然后每个人根据自己的具体配置，修改一份自用，但又不会将该配置文件提交到库中。

## 4. 参考文章

- [git 忽略不提交的文件 3 种情形](https://www.cnblogs.com/alice-fee/p/6757301.html)
