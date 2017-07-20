Ubuntu14.04下如何开启Mysql远程访问 

1.进入目录/etc/mysql下找到my.cnf，用vim编辑，找到my.cnf里面的
bind-address           = 127.0.0.1
注释掉这行改成如下：
#bind-address           = 127.0.0.1

2.然后用root登陆Mysql数据库 然后在执行

mysql> grant all on *.* to root@'%' identified by '123';
myslq> flush privileges;


就可以了。