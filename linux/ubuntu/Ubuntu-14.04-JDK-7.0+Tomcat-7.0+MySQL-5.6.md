
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
  <role rolename="manager-gui"/>
  <role rolename="manager-status"/>
  <role rolename="manager-script"/>
  <role rolename="manager-jmx"/>
  <role rolename="admin"/>
  <role rolename="admin-gui"/>
  <user username="tomcat" password="123456" roles="standard,manager-gui,manager-status,manager-script,admin-gui"/>

</tomcat-users>
```

注意：上面的 `password` 字段，请修改为你自己的密码。

### 3.1 远程访问权限

默认情况，`manager` 仅支持本地访问，如果需要远程访问，需要进行如下设置，文件地址如下：

```shell
$CATALINA_BASE/webapps/manager/META-INF/context.xml
```

编辑文件内容如下，允许两个 `IP` 地址段 `127.*.*.*` 和 `157.122.*.*` 的访问：

```xml
<?xml version="1.0" encoding="UTF-8"?>
<Context antiResourceLocking="false" privileged="true" >
  <Valve className="org.apache.catalina.valves.RemoteAddrValve"
         allow="127\.\d+\.\d+\.\d+|::1|0:0:0:0:0:0:0:1|157\.122\.\d+\.\d+|183\.240\.\d+\.\d+" />
  <Manager sessionAttributeValueClassNameFilter="java\.lang\.(?:Boolean|Integer|Long|Number|String)|org\.apache\.catalina\.filters\.CsrfPreventionFilter\$LruCache(?:\$1)?|java\.util\.(?:Linked)?HashMap"/>
</Context>
```

此外，新建一个 `manager.xml` 文件，如下：

```shell
$CATALINA_BASE/conf/Catalina/localhost/manager.xml
```

输入如下内容：

```xml
<?xml version="1.0" encoding="UTF-8"?>
<Context antiResourceLocking="false" privileged="true"
         docBase="${catalina.home}/webapps/manager">
  <Valve className="org.apache.catalina.valves.RemoteAddrValve"
         allow="127\.\d+\.\d+\.\d+|::1|0:0:0:0:0:0:0:1|157\.122\.\d+\.\d+|183\.240\.\d+\.\d+" />
</Context>
```

编辑重新读取配置的监视目录：

```shell
$CATALINA_BASE/conf/context.xml
```

修改为：

```xml
<?xml version="1.0" encoding="UTF-8"?>
<Context reloadable="true">

    <!-- Default set of monitored resources. If one of these changes, the    -->
    <!-- web application will be reloaded.                                   -->
    <WatchedResource>WEB-INF/web.xml</WatchedResource>
    <WatchedResource>${catalina.base}/conf/web.xml</WatchedResource>

    <!-- Uncomment this to disable session persistence across Tomcat restarts -->
    <!-- manager path -->
    <Manager pathname="${catalina.base}/webapps/manager" />
    <!-- manager path -->
</Context>
```
