
Ubuntu 14.04 搜索 apt 安装源
--------------------------------

`apt` 安装包查询功能的格式:

    $ apt-cache search ****

例如:

    $ apt-cache search mysql-server

安装 Mysql 5.5
-----------------

    $ sudo apt-get install mysql-server-5.5

安装 Mysql 5.6
-----------------

    $ sudo apt-get install mysql-server-5.6

Open JDK 7.0
---------------

搜索安装包:

    $ apt-cache search openjdk-7

`JDK 7` 安装命令:

    $ sudo apt-get install openjdk-7-jdk
    $ sudo apt-get install openjdk-7-doc
    $ sudo apt-get install openjdk-7-demo
    $ sudo apt-get install openjdk-7-dbg
    $ sudo apt-get install openjdk-7-source

也可以写成一句:

    $ sudo apt-get install openjdk-7-jdk openjdk-7-doc openjdk-7-demo openjdk-7-dbg openjdk-7-source

Tomcat 7.0
-------------

    $ apt-cache search tomcat7

可以查到以下结果:

    tomcat7 - Servlet and JSP engine
    tomcat7-admin - Servlet and JSP engine -- admin web applications
    tomcat7-common - Servlet and JSP engine -- common files
    tomcat7-docs - Servlet and JSP engine -- documentation
    tomcat7-examples - Servlet and JSP engine -- example web applications
    tomcat7-user - Servlet and JSP engine -- tools to create user instances

`Tomcat 7` 安装命令:

    $ sudo apt-get install tomcat7
    $ sudo apt-get install tomcat7-admin
    $ sudo apt-get install tomcat7-common
    $ sudo apt-get install tomcat7-docs
    $ sudo apt-get install tomcat7-examples
    $ sudo apt-get install tomcat7-user

也可以写成一句:

    $ sudo apt-get install tomcat7 tomcat7-admin tomcat7-common tomcat7-docs tomcat7-examples tomcat7-user
    
    

tomcat修改帐号密码：

找到配置文件“tomcat-users.xml”的路径，例如D:\soft\tomcat6\tomcat6\conf\tomcat-users.xml，用文本编辑器打开配置文件tomcat-users.xml，找到下面的信息并进行修改：

    <tomcat-users>  
    <!--  
      <role rolename="tomcat"/>  
      <role rolename="role1"/>  
      <user username="tomcat" password="tomcat" roles="tomcat"/>  
      <user username="both" password="tomcat" roles="tomcat,role1"/>  
      <user username="role1" password="tomcat" roles="role1"/>  
    -->  
      <role rolename="manager"/>  
      <user username="tomcat" password="tomcat" roles="manager"/>  
    </tomcat-users>  

注意要改成这样：
    <tomcat-users>  
    <!--  
      <role rolename="tomcat"/>  
      <role rolename="role1"/>  
      <user username="tomcat" password="tomcat" roles="tomcat"/>  
      <user username="both" password="tomcat" roles="tomcat,role1"/>  
      <user username="role1" password="tomcat" roles="role1"/>  
    -->  
        <role rolename="manager"/>
        <user username="tomcat" password="123" roles="manager"/>
        <role rolename="manager-gui"/>
        <user username="tomcat" password="123" roles="manager-gui"/>
    </tomcat-users>  
