首先来了解下yum命令
====================

yum= yellow dog updater, modified 主要功能更方便添加、删除、更新rpm包，自动解决依赖性问题，便于管理大量系统的更新问题

同时配置多个资源库（repository）简介的配置文件（/etc/yum.conf自动解决增加或删除rpm包时遇到的依赖性问题，方便保持rpm数据库的一致性）

yum安装，rpm -ivh yum-*.noarch.rpm在第一次启用yum之前要先导入系统的RPM-GPG-KEY

第一次使用yum管理软件时，yum会自动下载需要的headers放置在/var/cache/yum目录下



## rpm包更新

* yum check-update                          查看可以更新的软件包

* yum update                                更新所有的软件包

* yum update kernel                         更新指定的软件包

* yum upgrade                               大规模更新升级



## rpm包安装和删除

* yum install xxx[服务名]                    安装rpm包

* yum remove xxx[服务名]                      删除rpm包



## yum缓存信息

* yum clean packages                        清除缓存的rpm包文件

* yum clean headers                         清除缓存的rpm头文件

* yum clean old headers                     清除缓存中旧的头文件

* yum clean all                             清除缓存中旧的rpm头文件和包文件



## 查询软件包信息

* yum list                                   列出资源库中所有可以安装或更新的rpm包

* yum list firefox*                       列出资源库中可以安装、可以更新、已安装的指定rpm包

* yum list update                        列出资源库中可以更新的rpm包

* yum list installed                      列出所有已安装的rpm包

* yum list extras                         列出已安装但不包含在资源库中rpm包

## ps：通过网站下载安装的rpm包

* yum info                                列出资源库中所有可以安装或更新的rpm包信息

* yum info firefox*                       列出资源库可以安装或更新的指定的rpm的信息

* yum info update                         列出资源库中可以更新的rpm包信息

* yum info installed                      列出已安装的所有rpm包信息

* yum info extras                         列出已安装到时不包含在资源库中rpm包信息



* yum search firefox                      搜索匹配特定字符的rpm包

* yum provides firefox                    搜索包含特定文件的rpm包



## 删除文件夹实例：
* rm -rf /var/log/httpd/access
将会删除/var/log/httpd/access目录以及其下所有文件、文件夹






## java 的安装：

1.查询源：
yum search java | grep -i --color JDK

2.安装源  
sudo yum install java-1.7.0-openjdk.x86_64

3.删除已安装的软件 
sudo yum remove java-1.7.0-openjdk.x86_64

4.java的安装目录
/usr/lib/jvm/

手动方式安装java

1.复制java安装文件到 /usr/lib/jvm

sudo cp /root/downloads/jdk-7u80-linux-x64.tar.gz /usr/lib/jvm

2.解压
tar -zxvf jdk-8u74-linux-x64.gz#解压到当前目录,请把当前目录切换到jdk压缩包所在目录

3.删除安装包：
rm -f jdk-8u74-linux-x64.gz#删除文件  rm -rf 删除文件夹

4、配置jdk环境变量
vim /etc/profile

#java environment
export JAVA_HOME=/usr/lib/jvm/jdk1.7.0_80
export CLASSPATH=.:${JAVA_HOME}/jre/lib/rt.jar:${JAVA_HOME}/lib/dt.jar:${JAVA_HOME}/lib/tools.jar
export PATH=$PATH:${JAVA_HOME}/bin


5、生效jdk环境变量
source /etc/profile或 . /etc/profile

#如果后卸载OPENJDK，就必须再次使用生效命令
6、检查安装是否成功

java -version



## centos7下使用yum安装mysql

CentOS7的yum源中默认好像是没有mysql的。为了解决这个问题，我们要先下载mysql的repo源。

1. 下载mysql的repo源

$ wget http://repo.mysql.com/mysql-community-release-el7-5.noarch.rpm

2. 安装mysql-community-release-el7-5.noarch.rpm包


$ sudo rpm -ivh mysql-community-release-el7-5.noarch.rpm

安装这个包后，会获得两个mysql的yum repo源：/etc/yum.repos.d/mysql-community.repo，/etc/yum.repos.d/mysql-community-source.repo。

3. 安装mysql
?

$ sudo yum install mysql-server

根据步骤安装就可以了，不过安装完成后，没有密码，需要重置密码。

4. 重置密码

重置密码前，首先要登录
?

$ mysql -u root

登录时有可能报这样的错：ERROR 2002 (HY000): Can‘t connect to local MySQL server through socket ‘/var/lib/mysql/mysql.sock‘ (2)，原因是/var/lib/mysql的访问权限问题。下面的命令把/var/lib/mysql的拥有者改为当前用户：
?

$ sudo chown -R openscanner:openscanner /var/lib/mysql

然后，重启服务：
?

$ service mysqld restart

接下来登录重置密码：
?

$ mysql -u root
?
mysql > use mysql;
mysql > update user set password=password(‘123456‘) where user=‘root‘;
mysql> FLUSH PRIVILEGES;
mysql > exit;

5. 开放3306端口
?

$ sudo vim /etc/sysconfig/iptables

添加以下内容：
?

-A INPUT -p tcp -m state --state NEW -m tcp --dport 3306 -j ACCEPT

保存后重启防火墙：
?

$ sudo service iptables restart

这样从其它客户机也可以连接上mysql服务了。




## CentOS-7.0.中安装与配置Tomcat-7的方法

1.将apache-tomcat-7.0.29.tar.gz文件上传到/usr/local中执行以下操作：
sudo cp /root/downloads/apache-tomcat-7.0.76.tar.gz /usr/local

2.修改权限：
chmod 755 apache-tomcat-7.0.76.tar.gz

3.解压文件：
[root@linuxidc local]# tar -zxvf apache-tomcat-7.0.76.tar.gz // 解压压缩包
[root@linuxidc local]# rm -rf apache-tomcat-7.0.76.tar.gz // 删除压缩包
[root@linuxidc local]# mv apache-tomcat-7.0.76 tomcat


启动Tomcat

执行以下操作：

代码如下:

[root@linuxidc ~]# /usr/local/tomcat/bin/startup.sh //启动tomcat
Using CATALINA_BASE: /usr/local/tomcat
Using CATALINA_HOME: /usr/local/tomcat
Using CATALINA_TMPDIR: /usr/local/tomcat/temp
Using JRE_HOME: /usr/java/jdk1.7.0/jre
Using CLASSPATH: /usr/local/tomcat/bin/bootstrap.jar:/usr/local/tomcat/bin/tomcat-juli.jar

出现以上信息说明已成功启动。

防火墙开放8080端口

增加8080端口到防火墙配置中，执行以下操作：

[root@linuxidc ~]# vi + /etc/sysconfig/iptables

#增加以下代码

-A RH-Firewall-1-INPUT -m state --state NEW -m tcp -p tcp --dport 8080 -j ACCEPT

重启防火墙

[root@linuxidc java]# service iptables restart

检验Tomcat安装运行

通过以下地址查看tomcat是否运行正常：
http://192.168.15.231:8080/
看到tomcat系统界面，说明安装成功！

停止Tomcat

[root@linuxidc ~]#  /usr/local/tomcat/bin/shutdown.sh  //停止tomcat


Linux(CentOS7)设置Tomcat开机启动与内存设置 

1、/etc/rc.d/rc.local 文件最后加上下面俩行脚本。

export JAVA_HOME=/usr/lib/jvm/jdk1.7.0_80 
/usr/local/tomcat/bin/startup.sh start 

JAVA_HOME 是你jdk的安装路径

/usr/local/tomcat  是tomcat的安装目录


2、修改rc.local文件为可执行，如： chmod +x rc.local


3、重启机器：shutdown -r now
