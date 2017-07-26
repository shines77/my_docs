# ubuntu下安装nodejs #

安装node.js

1.通过 filezilla 把 本地的安装包文件node-v8.1.4-linux-x64.tar.xz 上传到服务器端 /usr/local/node 下。

2。进入 /usr/local/node 下修改 node-v8.1.4-linux-x64.tar.xz 安装包权限为 可读写可执行 
chmod 755 node-v8.1.4-linux-x64.tar.xz。

3。解压文件   
tar -xf node-v8.1.4-linux-x64.tar.xz。
4。进入解压后的目录
cd cd node-v8.1.4-linux-x64/bin

5.通过ln命令让  node和npm 可以在任何 路径下都可以执行。
ln -s /usr/local/node/node-v8.1.4-linux-x64/bin/node /usr/local/bin/node 
ln -s /usr/local/node/node-v8.1.4-linux-x64/bin/npm /usr/local/bin/npm

安装express

1.首先要使用Node.js的模块管理器npm安装Express：
sudo npm install -g express
sudo npm install -g express-generator

2.通过ln命令让  express 可以在任何 路径下都可以执行。
ln -s /usr/local/node/node-v8.1.4-linux-x64/bin/express /usr/bin/express 

3. cd 到home文件夹，并建立一个FHDemo文件夹，我们把例子放在这个目录下
express testapp
cd testapp
npm install
npm start



3.在浏览器访问http://10.110.200.141:3000/，可以看到我们的例子了。

4.但是关掉命令行，服务就停止了，但是当我们关闭终端之后，进程就将结束。 我们需要安装forever： 
sudo npm install -g forever

5.通过ln命令让  forever 可以在任何 路径下都可以执行。
ln -s /usr/local/node/node-v8.1.4-linux-x64/bin/forever /usr/bin/forever 

6.然后运行
forever start ./bin/www 

7. ok，即使我们关闭终端了，服务也会一直运行。(./bin/www 是package.json中start的脚本命令)

8.我们可以使用下面命令查看forever运行的程序： 
forever list

9.停止运行： forever stop 0//0代表前面[0],这是当前进程的ID

停止所有: forever stopall



ubuntu如何杀死进程 

一、查找进程

先用命令查询占用端口的某个进程
lsof -i:3000 

二、杀死进程
我们使用ps -ef命令之后，就会得到一些列进程信息，有进程pid什么的，如果你要杀死莫个进程的话，直接使用命令

sudo kill -9 pid
如：sudo kill -9 31379


ubuntu下查看进程端口

关键字: linux ubuntu # 查看所有打开的端口及服务名（注意这里显示的服务名只是标准端口对应的服务名，可能并不准确）

nmap localhost

测试其他主机端口是否打开

nmap ip -pport    -----------nmap 192.168.1.32 -p 22

# 查看哪些进程打开了指定端口port（对于守护进程必须以root用户执行才能查看到）

lsof -i:port

# 查看哪些进程打开了指定端口port，最后一列是进程ID（此方法对于守护进程作用不大）

netstat -nap|grep port

