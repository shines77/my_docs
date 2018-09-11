
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

push 的时候自动选择分支：（Since Git 1.7.11）

    $ git config --global push.default matching
    $ git config --global push.default simple

增加远程地址的时候带上密码也是可以的：（推荐）

    http://yourname:password@git.oschina.net/name/project.git

增加 `https` 远程仓库地址：

    $ git remote add origin http://yourname:password@git.oschina.net/name/project.git

    $ git config --global user.email "gz_shines@msn.com"
    $ git config --global user.name  "shines77"

当 git 账号的密码修改以后，重新设置仓库的用户名和email，下次pull就会重新要求输入密码啦：

    $ git config --global user.name "shines77"
    $ git config --global user.email "wokss@163.com"

    $ git pull
