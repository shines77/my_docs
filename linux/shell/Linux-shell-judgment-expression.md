# Linux shell 命令之判断表达式

## 1. 文件比较运算符

1. e filename

    如果 filename 存在，则为真，例如：[ -e /var/log/syslog ]

2. -d filename

    如果 filename 为目录，则为真，例如：[ -d /tmp/mydir ]

3. -f filename

    如果 filename 为常规文件，则为真，例如：[ -f /usr/bin/grep ]

4. -L filename

    如果 filename 为符号链接，则为真，例如：[ -L /usr/bin/grep ]

5. -r filename

    如果 filename 可读，则为真，例如：[ -r /var/log/syslog ]

6. -w filename

    如果 filename 可写，则为真，例如：[ -w /var/mytmp.txt ]

7. -x filename

    如果 filename 可执行，则为真，例如：[ -L /usr/bin/grep ]

8. filename1 -nt filename2

    如果 filename1 比 filename2 新，则为真，例如：[ /tmp/install/etc/services -nt /etc/services ]

9. filename1 -ot filename2

    如果 filename1 比 filename2 旧，则为真，例如：[ /boot/bzImage -ot arch/i386/boot/bzImage ]

## 2. 字符串比较运算符

（请注意引号的使用，这是防止空格扰乱代码的好方法）

1. -z string

    如果 string 长度为零，则为真，例如：[ -z "$myvar" ]

2. -n string

    如果 string 长度非零，则为真，例如：[ -n "$myvar" ]

3. string1 = string2

    如果 string1 与 string2 相同，则为真，例如：["$myvar" = "one two three"]

4. string1 != string2

    如果 string1 与 string2 不同，则为真，例如：["$myvar" != "one two three"]

## 3. 算术比较运算符

1. num1 -eq num2

    等于真，例如：[ 3 -eq $mynum ]

2. num1 -ne num2

    不等于真，例如：[ 3 -ne $mynum ]

3. num1 -lt num2

    小于真，例如：[ 3 -lt $mynum ]

4. num1 -le num2

    小于或等于真，例如：[ 3 -le $mynum ]

5. num1 -gt num2

    大于真，例如：[ 3 -gt $mynum ]

6. num1 -ge num2

    大于或等于真，例如：[ 3 -ge $mynum ]

## 4. 逻辑运算符

1. ! express

     逻辑非，条件表达式的相反，例如：

    ```bash
    if [ ! 表达式 ]; then
    if [ ! -d $num ]; then   ## 如果$num目录不存在
    ```

2. express1 -a express2

     逻辑与，条件表达式的并列，例如：

     ```bash
     if [ 表达式1 –a 表达式2 ]
     ```

3. express1 -o express2

     逻辑或，条件表达式的或，例如：

     ```bash
     if [ 表达式1 –o 表达式2 ]
     ```

## 5. 查看磁盘、文件大小 

1. df -h

    查看磁盘占用情况

2. du -sm ./*

    查看当前目录下文件大小，单位 M

## 6. 参考文章

- [linux shell 指令 诸如-d, -f, -e之类的判断表达式简介](https://www.cnblogs.com/zrmw/p/9625727.html)
