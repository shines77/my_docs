
# 在 Ubuntu 14.04 上安装和配置 Tomcat 8.0

## 1. 下载

### 1.1. 官网页面

官网的下载页面：

[http://tomcat.apache.org/download-80.cgi](http://tomcat.apache.org/download-80.cgi)

可以选择下载 `tar.gz` 或者 `zip` 格式。

### 1.2. 下载文件

这里选择 `tar.gz` 格式，下载 `tomcat8` 源码包：

```shell
cd /home/web
mkdir tomcat
wget -c http://mirrors.shu.edu.cn/apache/tomcat/tomcat-8/v8.5.38/bin/apache-tomcat-8.5.38.tar.gz
```

**其他方式**

如果你不想在服务器上下载 `Tomcat8` 源码包文件，也还有一种方法，就是使用 `rz`（上传）或 `sz`（下载）命令使用 `SSH` 终端上传文件到服务器上（在 `XShell` 终端上是支持的，`SecureCRT` 和 `Xmanager` 也支持，其他终端不清楚）。

当然，你也可以使用 `WinSCP` 这样的支持 `SFTP` 协议的软件把文件上传到服务器上。

**使用 rz 命令上传文件**

首先，你得安装 `lrzsz`：

```
apt-get install lrzsz -y
```

```shell
cd /home/web
mkdir tomcat
rz
```

输入 `rz` 命令后，会弹出对话框让你选择要上传的文件，选中要上传的文件即可。

### 1.3. 解压文件

这里打算把 `tomcat8` 放在 `/usr/lib` 下面，先解压。

```shell
tar -zxvf apache-tomcat-8.5.38.tar.gz -C /usr/lib

cd /usr/lib
mv apache-tomcat-8.5.38 tomcat8
```

这里 "`tar -zxvf`" 的参数含义：

`-z` 即 "`--gzip, --gunzip`"，表示指定的格式是 `gzip` 格式。`-x` 即 "`--extract`"，表示是解压文件。`-f` 即 "`--file=ARCHIVE`"，表示后面的参数是指定的解压文件的路径和文件名。`-v` 即 "`--verbo`"，表示显示已经出来的文件列表信息。`-C` 即 "`--directory=DIR`"，表示解压或压缩的文件夹。

## 2. 配置

有两种配置方式，推荐使用第一种。

### 2.1. 修改全局配置文件

#### 2.1.1. 修改 /etc/rc.local
    
在 `/etc/rc.local` 文件的最后一行 `exit 0` 之前添加如下内容：

```bash
vim /etc/rc.local

export JAVA_HOME=/usr/lib/jvm/java-8-oracle
export JRE_HOME=$JAVA_HOME/jre
export CLASSPATH=.:$JAVA_HOME/lib/dt.jar:$JAVA_HOME/lib/tools.jar:$JRE_HOME/lib/rt.jar
export TOMCAT_HOME=/usr/lib/tomcat8
export CATALINA_HOME=$TOMCAT_HOME
export PATH=$PATH:$JAVA_HOME/bin

exit 0
```

注：可根据你的环境修改相应的配置，比如 `JDK` 的路径（`JAVA_HOME`）。如果你不知道 `JDK` 路径设置在哪，可以输入 `echo $JAVA_HOME` 查询当前 `JDK` 的安装路径。也可以使用 `which java` 命令，查找 `java` 的可执行文件的路径，如果该执行文件做了 `软连接`，顺藤摸瓜一级一级找下去，同样可以找到 `JDK` 的具体安装路径。

保存文件后，需要重启服务器才能生效。

#### 2.1.2. 修改 /etc/profile

在 `/etc/profile` 文件任意地方添加如下内容：

```bash
vim /etc/profile

export JAVA_HOME=/usr/lib/jvm/java-8-oracle
export JRE_HOME=$JAVA_HOME/jre
export CLASSPATH=.:$JAVA_HOME/lib/dt.jar:$JAVA_HOME/lib/tools.jar:$JRE_HOME/lib/rt.jar
export TOMCAT_HOME=/usr/lib/tomcat8
export CATALINA_HOME=$TOMCAT_HOME
export PATH=$PATH:$JAVA_HOME/bin
```

保存文件后，需要重新登陆 `SSH` 终端才能生效。

### 2.2. 修改 Tomcat8 配置文件

修改启动和关闭文件：`startup.sh` 和 `shutdown.sh` 文件。

```shell
cd /usr/lib/tomcat8
vim ./bin/startup.sh
```

在 `./bin/startup.sh` 文件的最后一句的前面，添加下列内容，如下所示：

```bash
JAVA_HOME=/usr/lib/jvm/java-8-oracle
JRE_HOME=$JAVA_HOME/jre
CLASSPATH=.:$JAVA_HOME/lib/dt.jar:$JAVA_HOME/lib/tools.jar:$JRE_HOME/lib/rt.jar
TOMCAT_HOME=/usr/lib/tomcat8
CATALINA_HOME=$TOMCAT_HOME
PATH=$PATH:$JAVA_HOME/bin

exec "$PRGDIR"/"$EXECUTABLE" start "$@"
```

注：可根据你的环境修改相应的配置，比如 `JDK` 的路径（`JAVA_HOME`）。如果你不知道 `JDK` 路径设置在哪，可以输入 `echo $JAVA_HOME` 查询当前 `JDK` 的安装路径。

类似的，在 `./bin/shutdown.sh` 文件里也添加同样的内容（也是添加到该文件最后一句的前面）：

```shell
vim ./bin/shutdown.sh

# 添加内容同上，不再敖述
```

## 3. 启动/关闭

### 3.1. 启动 Tomcat8

```shell
cd /usr/lib/tomcat8
./bin/startup.sh
```

如果启动成功，会显示如下结果：

```shell
Using CATALINA_BASE:   /usr/lib/tomcat8
Using CATALINA_HOME:   /usr/lib/tomcat8
Using CATALINA_TMPDIR: /usr/lib/tomcat8/temp
Using JRE_HOME:        /usr/lib/jvm/java-8-oracle
Using CLASSPATH:       /usr/lib/tomcat8/bin/bootstrap.jar:/usr/lib/tomcat8/bin/tomcat-juli.jar
Tomcat started.
```

此外，还可以用别的方式检查是否启动成功。

第一个方式，可以检查 `Tomcat8` 的端口是否已经打开，默认端口是 `8080`。

```shell
netstat -tpln | grep 8080
netstat -apln  # 查看所有连接
```

如果显示如下结果，说明 `8080` 端口打开成功。

```shell
tcp     0     0 0.0.0.0:8080      0.0.0.0:*       LISTEN
```

第二个方式就是打开浏览器访问一下看看。比如：

```
http://localhost:8080/
或者
http://{你的服务器的域名或IP地址}:8080/
```

### 3.2. 关闭 Tomcat8

```shell
cd /usr/lib/tomcat8
./bin/shutdown.sh
```

执行关闭脚本后，显示的信息不再敖述。

