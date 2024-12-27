# IDEA 和 SpringBoot 文章

- [IDEA2021.2 安装及配置教程（最新保姆级）](https://blog.csdn.net/xikaifeng/article/details/120550307)

- [SpringBoot配置文件&YAML配置注入（详解）](https://blog.csdn.net/qq_45173404/article/details/108693030)

- [IDEA 使用教程（图文讲解）](https://www.quanxiaoha.com/idea/idea-tutorial.html)

- [Spring Boot 最全面的入门教程](https://springdoc.cn/spring-boot-start/)

- [Spring Boot 参考文档 3.3.5](https://docs.springframework.org.cn/spring-boot/documentation.html)

- [面试必问的40个SpringBoot面试题！](https://cloud.tencent.com/developer/article/2110061)

- [SpringBoot面试题及答案（最新50道大厂版，持续更新）](https://blog.csdn.net/yanpenglei/article/details/134989397)

    Spring Boot的自动配置原理基于 @SpringBootApplication 注解，它是 @Configuration、@EnableAutoConfiguration 和 @ComponentScan 的组合。

- [MyBatis 教程](https://www.ddkk.com/category/orm/mybatis/1/index.html)

    虽然不是 Spring Boot + MyBaties 的教程，但是对于学习 MyBaties 还是有帮助的，比较完整。

- [SpringBoot集成Mybatis保姆级教程（完整版）](https://blog.csdn.net/2302_78411932/article/details/137083650)

    相对比较完整的介绍，不过可能也有一点点小问题。

- [springboot整合mybatis（详解）](https://blog.csdn.net/zhenghuishengq/article/details/109510128)

    有点小问题，但可以借鉴一下。

- [MySQL完整安装和配置(保姆级教程)](https://blog.csdn.net/langjiaohjiopji/article/details/143512551)

    ```bash
    net start mysql80
    net stop mysql80

    mysql [-h 127.0.0.1] [-P 3306] -u root -p

    参数：
        -h : MySQL服务所在的主机IP
        -P : MySQL服务端口号， 默认3306
        -u : MySQL数据库用户名
        -p ： MySQL数据库用户名对应的密码
    ```

    ```sql
    CREATE TABLE `user` (
      `id` int(11) NOT NULL AUTO_INCREMENT,
      `username` varchar(255) COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '账号',
      `nickname` varchar(255) COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '昵称',
      `password` varchar(32)  COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '密码',
      PRIMARY KEY (`id`)
    ) ENGINE=InnoDB AUTO_INCREMENT=2 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

    insert into user (id, username, nickname, password) values (DEFAULT, 'shines77', 'Guozi', '123456');

    CREATE TABLE `book` (
      `id` int(11) NOT NULL AUTO_INCREMENT,
      `title` varchar(255) COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '标题',
      `author` varchar(255) COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '作者',
      PRIMARY KEY (`id`)
    ) ENGINE=InnoDB AUTO_INCREMENT=2 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
    ```

    ```sql
    ALTER USER 'root'@'localhost' IDENTIFIED BY '123456';
    ALTER USER 'sa'@'%' IDENTIFIED BY '123456';
    ```

    ```sql
    use mysql;
    select host, user from mysql.user;

    +-----------+------------------+
    | host      | user             |
    +-----------+------------------+
    | %         | sa               |
    | localhost | mysql.infoschema |
    | localhost | mysql.session    |
    | localhost | mysql.sys        |
    | localhost | root             |
    +-----------+------------------+
    5 rows in set (0.00 sec)
    ```

    修改用户的 host ：

    ```sql
    UPDATE mysql.user SET host = '192.168.1.100' WHERE user = 'myuser';
    FLUSH PRIVILEGES;
    ```

    创建一个允许从任何主机连接的用户：

    ```sql
    CREATE USER 'myuser'@'%' IDENTIFIED BY 'mypassword';
    GRANT ALL PRIVILEGES ON mydatabase.* TO 'myuser'@'%';
    FLUSH PRIVILEGES;
    ```
- [使用CGLIB来创建代理，而不是JDK动态代理](https://copilot.tencent.com/chat)

    错误信息

    ```bash
    ***************************
    APPLICATION FAILED TO START

    ***************************

    Description:

    The bean 'userMapper' could not be injected because it is a JDK dynamic proxy

    The bean is of type 'jdk.proxy2.$Proxy123' and implements:

    com.i77studio.mapper.UserMapper

    Expected a bean of type 'com.i77studio.mapper.UserMapperImpl' which implements:

    com.i77studio.mapper.UserMapper
    ```

    应用程序启动失败的原因是因为Spring尝试注入一个JDK动态代理类型的bean，而期望的是一个具体的实现类bean。

    错误信息表明，Spring容器中存在一个类型为`jdk.proxy2.$Proxy123`的bean，它实现了`com.i77studio.mapper.UserMapper`接口。但是，Spring期望的是一个类型为`com.i77studio.mapper.UserMapperImpl`的bean，同样实现了`com.i77studio.mapper.UserMapper`接口。

    这个问题通常发生在使用MyBatis时，因为MyBatis默认使用JDK动态代理来创建Mapper接口的实现。而Spring在某些情况下可能期望一个具体的实现类而不是代理类。

    要解决这个问题，您可以尝试以下方法：

    1. **使用CGLIB代理**：在MyBatis的配置中，您可以指定使用CGLIB来创建代理，而不是JDK动态代理。这可以通过在`application.properties`或`application.yml`文件中设置以下属性来实现：

    ```properties
    mybatis.configuration.proxy-target-class=true
    ```

    或者在`application.yml`中：

    ```yaml
    mybatis:
      configuration:
        proxy-target-class: true
    ```

    2. **使用@Bean注解**：如果您希望Spring管理一个具体的实现类bean，您可以在配置类中使用`@Bean`注解来显式定义这个bean。例如：

    ```java
    @Configuration
    public class MyBatisConfig {

        @Bean
        public UserMapper userMapper() {
            return new UserMapperImpl();
        }
    }
    ```

    3. **检查@Autowired注解的使用**：确保您没有在接口上直接使用`@Autowired`注解，而是应该在实现类或者使用这些接口的服务类上使用。

    4. **检查Spring Boot版本**：确保您使用的Spring Boot版本与MyBatis以及其它相关依赖兼容。

    尝试上述方法之一应该能够解决您遇到的问题。如果问题仍然存在，请检查您的配置和依赖项，确保它们之间没有冲突。

