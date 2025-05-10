
# git-push 时不需要输入密码，自动保存密码 #

参考自：

[http://my.oschina.net/amath0312/blog/389775]()

[http://my.oschina.net/u/244918/blog/393642]()

记住密码（默认只保存 15 分钟）:

    $ git config --global credential.helper cache

如果想自己设置时间，可以这样做：

    $ git config credential.helper 'cache --timeout=3600'

永久存储密码(无时间限制，不失效，强烈推荐！)：

    $ git config --global credential.helper store

清除存储的凭证缓存：

    $ git credential-cache exit

push 的时候自动选择分支：（Since Git 1.7.11）

    $ git config --global push.default matching
    $ git config --global push.default simple

增加远程地址的时候带上密码也是可以的：（推荐）

    http://yourname:password@git.oschina.net/name/project.git

增加 `https` 远程仓库地址：

    $ git remote add origin http://yourname:password@git.oschina.net/name/project.git

    $ git config --global user.name "shines77"
    $ git config --global user.email "gz_shines@msn.com"

当 git 账号的密码修改以后，重新设置仓库的用户名和email，下次pull就会重新要求输入密码啦：

    $ git config --global user.name "shines77"
    $ git config --global user.email "wokss@163.com"

    $ git config --local user.name "shines77"
    $ git config --local user.email "wokss@163.com"

    $ git pull

显示 config 信息，该信息可以通过 `vim .git/config` 命令编辑：

    $ git config --global --list

    credential.helper=store
    user.name=shines77
    user.email=gz_shines@msn.com

本地仓库配置：

    $ git config --local --list

    credential.helper=store
    user.email=wokss@163.com
    user.name=shines77
    push.default=matching
    core.repositoryformatversion=0
    core.filemode=true
    core.bare=false
    core.logallrefupdates=true
    remote.origin.url=https://git.oschina.net/shines77/jlang.git
    remote.origin.fetch=+refs/heads/*:refs/remotes/origin/*
    branch.master.remote=origin
    branch.master.merge=refs/heads/master
