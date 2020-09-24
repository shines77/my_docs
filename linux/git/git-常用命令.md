
显示 global 配置的内容:

git config --global --list   // 显示 global 配置的内容

$ vim ~/.gitconfig             // 等价于上面那一句

显示 config 信息，该信息可以通过 vim .git/config 命令编辑：

git config --list

设置全局设置的用户名和email:

git config --global user.name=shines77
git config --global user.email=gz_shines@msn.com

永久存储密码(无时间限制，不失效，强烈推荐！)：

git config --global credential.helper store

git 拉取远程分支:

git pull -v --progress "origin"

git push --progress "origin" master:master

git 读取远端仓库地址：

git remote get-url --all origin

git 修改远程仓库地址：

git remote set-url origin https://github.com/shines77/ceph.git

git remote set-url origin http://192.168.3.225:3000/shines77/fitdep.git
git remote set-url origin http://192.168.3.225:3000/shines77/fitpxe.git

Initialize by fitdep_init.py.

添加远程仓库地址:

git remote add my_repo https://github.com/shines77/ceph.git


远程仓库 my_repo 改名为 develop:

git remote rename my_repo develop

查询所有的远程仓库信息:

git remote -v

查看最后提交历史:

git log --pretty --graph --decorate
git log --oneline --graph --decorate

git log -p -10                        # 显示最后更新的10条提交历史

git log --tags=juno-eol               # 显示标签juno-eol的提交历史

git log --branches=stable/juno-eol    # 显示分支stable/juno-eol的提交历史

git log -author=abcd                  # 显示abcd用户的提交历史

# 1.查看所有分支

> git branch -a

# 2.查看当前使用分支(结果列表中前面标*号的表示当前使用分支)

> git branch
 
# 3.切换分支

> git checkout 分支名

git checkout -B stable/juno                           # 在本地创建一个叫stable/juno的新分支
git checkout -b stable/juno --track juno-eol          # 拉取远程分支juno-eol,并且把分支重命名为stable/juno


初始化子模块:

git submodule init
git submodule update

或者

git submodule update --init --recursive

添加子模块:

git submodule add <url> <path>

git submodule add https://github.com/shines77/jstd.git deps/jstd

(从当前的远程仓库)更新子模块:

git submodule sync --recursive

(从子模块官方远程仓库)更新子模块:

git submodule update --remote


删除子模块

有时子模块的项目维护地址发生了变化，或者需要替换子模块，就需要删除原有的子模块。

删除子模块较复杂，步骤如下：

    rm -rf 子模块目录 删除子模块目录及源码
    vi .gitmodules 删除项目目录下.gitmodules文件中子模块相关条目
    vi .git/config 删除配置项中子模块相关条目
    rm .git/module/* 删除模块下的子模块目录，每个子模块对应一个目录，注意只删除对应的子模块目录即可

执行完成后，再执行添加子模块命令即可，如果仍然报错，执行如下：

git rm --cached 子模块名称

完成删除后，提交到仓库即可。
