
# Ubuntu 14.04 安装 JDK 7.0 + Tomcat 7.0 + MySQL 5.6

## 1. MySQL

`apt` 安装包查询功能的格式:

```shell
apt-cache search ****
```

例如:

```shell
apt-cache search mysql-server
```

查询的结果是：

```shell
mysql-server - MySQL database server (metapackage depending on the latest version)
mysql-server-5.5 - MySQL database server binaries and system database setup
mysql-server-core-5.5 - MySQL database server binaries
auth2db - Powerful and eye-candy IDS logger, log viewer and alert generator
mariadb-server-5.5 - MariaDB database server binaries
mariadb-server-core-5.5 - MariaDB database core server files
mysql-server-5.6 - MySQL database server binaries and system database setup
mysql-server-core-5.6 - MySQL database server binaries
percona-xtradb-cluster-server-5.5 - Percona Server database server binaries
torrentflux - web based, feature-rich BitTorrent download manager
```

我们可以看到 `Ubuntu 14.04` 官方的安装源上 `mysql` 只有两个版本，即 `5.5` 和 `5.6` 。

还可以尝试 `mysql-`，`mysql-server` 等关键字。

### 1.1. 安装 Mysql 5.5

```shell
sudo apt-get install mysql-server-5.5
```

### 1.2. 安装 Mysql 5.6

```shell
sudo apt-get install mysql-server-5.6
```

## 2. Open JDK 7.0

搜索安装包:

```shell
apt-cache search openjdk-7
```

`JDK 7` 安装命令:

```shell
sudo apt-get install openjdk-7-jdk
sudo apt-get install openjdk-7-doc
sudo apt-get install openjdk-7-demo
sudo apt-get install openjdk-7-dbg
sudo apt-get install openjdk-7-source
```

也可以写成一句:

```shell
sudo apt-get install openjdk-7-jdk openjdk-7-doc openjdk-7-demo openjdk-7-dbg openjdk-7-source
```

## 3. Tomcat 7.0

查询 `Tomcat 7.0` 的安装包：

```shell
apt-cache search tomcat7
```

可以查到以下结果:

```shell
tomcat7 - Servlet and JSP engine
tomcat7-admin - Servlet and JSP engine -- admin web applications
tomcat7-common - Servlet and JSP engine -- common files
tomcat7-docs - Servlet and JSP engine -- documentation
tomcat7-examples - Servlet and JSP engine -- example web applications
tomcat7-user - Servlet and JSP engine -- tools to create user instances
```

`Tomcat 7` 安装命令:

```shell
sudo apt-get install tomcat7
sudo apt-get install tomcat7-admin
sudo apt-get install tomcat7-common
sudo apt-get install tomcat7-docs
sudo apt-get install tomcat7-examples
sudo apt-get install tomcat7-user
```

也可以写成一句:

```shell
sudo apt-get install tomcat7 tomcat7-admin tomcat7-common tomcat7-docs tomcat7-examples tomcat7-user
```

`tomcat` 修改管理帐号和密码：

找到配置文件 “`tomcat-users.xml`” 的路径，例如 `\tomcat7\conf\tomcat-users.xml`，用文本编辑器打开配置文件 `tomcat-users.xml`，找到下面的信息并进行修改：

```xml
<tomcat-users>

// 前面省略 ...

<!--
    <role rolename="tomcat"/>
    <role rolename="role1"/> 
    <user username="tomcat" password="tomcat" roles="tomcat"/>
    <user username="both" password="tomcat" roles="tomcat,role1"/>
    <user username="role1" password="tomcat" roles="role1"/>
-->
</tomcat-users>
```

注意要改成这样：

```xml
<tomcat-users>

// 前面省略 ...

<!--
    <role rolename="tomcat"/>
    <role rolename="role1"/> 
    <user username="tomcat" password="tomcat" roles="tomcat"/>
    <user username="both" password="tomcat" roles="tomcat,role1"/>
    <user username="role1" password="tomcat" roles="role1"/>
-->
    <role rolename="manager"/>
    <user username="tomcat" password="123456" roles="manager"/>
    <role rolename="manager-gui"/>
    <user username="tomcat" password="123456" roles="manager-gui"/>
    
</tomcat-users>
```

注意：上面的 `password` 字段，请修改为你自己的密码。
