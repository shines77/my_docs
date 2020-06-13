
显示 global 配置的内容:

$ git config --global --list   // 显示 global 配置的内容

$ vim ~/.gitconfig             // 等价于上面那一句

显示 config 信息，该信息可以通过 vim .git/config 命令编辑：

$ git config --list

设置全局设置的用户名和email:

$ git config --global user.name=shines77
$ git config --global user.email=gz_shines@msn.com

永久存储密码(无时间限制，不失效，强烈推荐！)：

$ git config --global credential.helper store

git 拉取远程分支:

$ git pull -v --progress "origin"

$ git push --progress "origin" master:master

git 读取远端仓库地址：

$ git remote get-url --all origin

git 修改远程仓库地址：

$ git remote set-url origin https://github.com/shines77/ceph.git

$ git remote set-url origin http://192.168.3.225:3000/shines77/fitdep.git
$ git remote set-url origin http://192.168.3.225:3000/shines77/fitpxe.git

Initialize by fitdep_init.py.

添加远程仓库地址:

$ git remote add my_repo https://github.com/shines77/ceph.git


远程仓库 my_repo 改名为 develop:

$ git remote rename my_repo develop

查询所有的远程仓库信息:

$ git remote -v

查看最后提交历史:

$ git log --pretty --graph --decorate
$ git log --oneline --graph --decorate

$ git log -p -10                        # 显示最后更新的10条提交历史

$ git log --tags=juno-eol               # 显示标签juno-eol的提交历史

$ git log --branches=stable/juno-eol    # 显示分支stable/juno-eol的提交历史

$ git log -author=abcd                  # 显示abcd用户的提交历史

$ git checkout -B stable/juno                           # 在本地创建一个叫stable/juno的新分支
$ git checkout -b stable/juno --track juno-eol          # 拉取远程分支juno-eol,并且把分支重命名为stable/juno


初始化子模块:

$ git submodule init
$ git submodule update --init --recursive
