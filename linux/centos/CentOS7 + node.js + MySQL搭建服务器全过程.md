# CentOS7 + node.js + MySQL搭建服务器全过程

## 登录服务器

`ssh root@10.110.200.141` 之后输入密码即可。

## 安装git

`sudo yum install git`

## 安装nodejs

* 直接下载

  cd 到要放下载文件的目录。我这边是在 /usr/local/src 下新建了目录tools。

  wget https://nodejs.org/dist/v8.1.3/node-v8.1.3.tar.gz

  由于网络原因，服务器一直在连接，所以我采用的先在mac电脑上下载下来，然后传输到服务器的方式。

* 下载之后再传输

  一：进入官网下载页面：https://nodejs.org/en/download/ 选择linux版本下载到本地

  二：下载FileZilla，一种用于和服务器之间传输文件的图片界面软件，比较直观。
（命令行玩家可以使用scp命令：`scp ~/local/file user@remote:~/file` 左边参数是你 Mac 电脑里想要上传的文件路径，右边是服务器上的路径。如果没配置 ssh key，会提示你输入密码，照办就是）

  三：把下载下来的node包传入到服务器目录下。

* 解压

  cd到相应目录下执行 `tar xvf node-v8.1.3-linux-x64.tar.xz` (我执行这个命令的用tar zxvf时候报错，不太明白zxvf代表什么意思。。。。后面查了文档再补充)

  解压后在bin文件夹中已经存在node以及npm，  `cd node-v0.10.28-linux-x64/bin`, 然后执行`./node -v`，可以看到命令可以执行，只不过不是全局的，需要手动设置为全局命令：

  `ln -s /usr/src/tools/node-v8.1.3-linux-x64/bin/node /usr/local/bin/node`
  `ln -s /usr/src/tools/node-v8.1.3-linux-x64/bin/npm /usr/local/bin/npm`

  cd 到根目录，执行`node -v` 和 `npm -v`, 可以正常执行，至此nodejs安装完成。

* 测试

  首先要使用Node.js的模块管理器npm安装Express：

  `sudo npm install -g express`

	 这还不行，还需要

	`sudo npm install -g express-generator`

  cd 到home文件夹，并建立一个FHDemo文件夹，我们把例子放在这个目录下

  ```
  express testapp
  cd testapp
  npm install
  npm start
  ```

  在浏览器访问http://10.110.200.141:3000/，可以看到我们的例子了。

  但是关掉命令行，服务就停止了，但是当我们关闭终端之后，进程就将结束。 我们需要安装forever：
  `sudo npm install -g forever`

  然后运行`forever start ./bin/www`。 ok，即使我们关闭终端了，服务也会一直运行。(./bin/www 是package.json中start的脚本命令)

  我们可以使用下面命令查看forever运行的程序：
  `forever list`

  停止运行：
  `forever stop 0 `//0代表前面[0],这是当前进程的ID

  停止所有:
  `forever stopall`

## 安装MySQL

  查看可用版本
  `yum list | grep mysql`

  在centOS 7中不能使用yum -y install mysql mysql-server mysql-devel安装，这样会默认安装mysql的分支mariadb。
  MariaDB数据库管理系统是MySQL的一个分支，主要由开源社区在维护，采用GPL授权许可 MariaDB的的是完全兼容MySQL，包括API和命令行，使之能轻松成为MySQL的代替品。

  正确的安装方法:
  `yum -y install mysql-community-server`

  MySQL基础配置:

  ```
  systemctl enable mysqld //添加到开机启动
  systemctl start mysqld //启用进程
  mysql_secure_installation
  ```

  创建user用户:
  `CREATE USER demo IDENTIFIED BY “123456” `

  配置远程连接:
  `mysql>GRANT ALL PRIVILEGES ON shandong.* TO 'demo'@'%'WITH GRANT OPTION`

  赋予任何主机访问数据的权限，也可以如下操作:
  `GRANT ALL PRIVILEGES ON shandong.* TO 'demo'@'%'IDENTIFIED BY '123456' WITH GRANT OPTION; `

  修改生效:
  `mysql>FLUSH PRIVILEGES `

  退出MySQL服务器:
  `mysql>EXIT `

## 参考资料：

  【链接】远程连接mysql授权方法详解 http://www.jb51.net/article/31902.htm
