# Git 推送和删除远程标签

（ 转载自：[http://ihacklog.com/post/how-to-push-and-delete-a-remote-git-tag.html](http://ihacklog.com/post/how-to-push-and-delete-a-remote-git-tag.html) ）

## 1. git tag

事实上 `Git` 的推送和删除远程标签命令是相同的，删除操作实际上就是推送空的源标签 `refs`：

```bash
git push origin 标签名
```

相当于

```bash
git push origin refs/tags/源标签名:refs/tags/目的标签名
```

`git push` 文档中有解释：

```bash
tag <<tag>> means the same as refs/tags/<tag>:refs/tags/<tag>.
Pushing an empty <src> allows you to delete the <dst> ref from the remote repository.
```

推送标签：

```bash
git push origin 标签名
```

删除本地标签：

```bash
git tag -d 标签名
```

删除远程标签：

```bash
git push origin :refs/tags/标签名

git push origin :refs/tags/protobuf-2.5.0rc1
```

其他本地操作：

```bash
# 打标签
git tag -a v1.1.4 -m "tagging version 1.1.4"

# 删除本地仓库标签
git tag -d v1.1.4

# 列出标签
git tag
```

## 2. 参考文章

* [http://nathanhoad.net/how-to-delete-a-remote-git-tag](http://nathanhoad.net/how-to-delete-a-remote-git-tag)

* [http://linux.die.net/man/1/git-push](http://linux.die.net/man/1/git-push)
