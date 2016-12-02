如何卸载 MySQL
==================

先停止 `mysql` 服务，卸载组件(如果是其他版本改一下版本号试试)：

```shell
$ sudo service mysql stop

$ sudo apt-get purge mysql-client-5.6
$ sudo apt-get purge mysql-community-client-5.6
$ sudo apt-get purge mysql-server-5.6
$ sudo apt-get purge mysql-community-server-5.6
$ sudo apt-get autoremove
$ sudo apt-get autoclean
```

`mysql` 配置目录：

```shell
/etc/mysql/
```

`mysql` 数据库目录：

```shell
/var/lib/mysql/
```

`mysql` 文档目录：

```shell
/usr/share/mysql/
```

还可以用 “`locate mysql`” 以及 “`find / -name "mysql"`” 语句来查看跟 `mysql` 相关的文件和目录。

`dump` 当前的数据库到 `*.sql` 文件：

```shell
$ sudo mysqldump -u root -p --add-drop-table --routines --events --all-databases --force > data-for-downgrade.sql
```

http://192.168.3.163/latest/meta-data/public-hostname

[EL Warning]: 2016-11-22 11:59:22.059--ServerSession(1011246983)-- The reference column name [resource_type_id] mapped on the element [field permissions] does not correspond to a valid id or basic field/column on the mapping reference. Will use referenced column name as provided.
[EL Info]: 2016-11-22 11:59:22.412--ServerSession(1011246983)-- EclipseLink, version: Eclipse Persistence Services - 2.6.2.v20151217-774c696
[EL Info]: 2016-11-22 11:59:22.859--ServerSession(1011246983)-- /file:/usr/lib/ambari-server/ambari-server-2.2.2.0.460.jar_ambari-server_nonJtaDataSource=1833788346_url=jdbc:mysql://localhost:3306/ambari_user=ambari login successful


