# Linux shell 命令之判断表达式

## 1. 文件比较运算符

1. -e filename

    如果 filename （文件或目录）存在，则为真，例如：[ -e /var/log/syslog ]

2. -a filename

    如果 filename 为文件，则为真，例如：[ -a /tmp/log/syslog ]

3. -d filename

    如果 filename 为目录，则为真，例如：[ -d /tmp/mydir ]

4. -f filename

    如果 filename 为常规文件，则为真，例如：[ -f /usr/bin/grep ]

5. -L filename

    如果 filename 为符号链接，则为真，例如：[ -L /usr/bin/grep ]

6. -r filename

    如果 filename 可读，则为真，例如：[ -r /var/log/syslog ]

7. -w filename

    如果 filename 可写，则为真，例如：[ -w /var/mytmp.txt ]

8. -x filename

    如果 filename 可执行，则为真，例如：[ -L /usr/bin/grep ]

9. filename1 -nt filename2

    如果 filename1 比 filename2 新，则为真，例如：[ /tmp/install/etc/services -nt /etc/services ]

10. filename1 -ot filename2

    如果 filename1 比 filename2 旧，则为真，例如：[ /boot/bzImage -ot /arch/i386/boot/bzImage ]

11. filename1 -ef filename2

    如果 filename1 和 filename2 一样新，则为真，例如：[ /boot/bzImage -ef /arch/i386/boot/bzImage ]

12. 其他不常用的

    ```bash
    [ -b FILE ]：如果 FILE 存在，且是一个块文件，则返回为真。
    [ -c FILE ]：如果 FILE 存在，且是一个字符文件，则返回为真。
    [ -g FILE ]：如果 FILE 存在，且设置了 SGID 则返回为真。
    [ -h FILE ]：如果 FILE 存在，且是一个符号符号链接文件，则返回为真。（该选项在一些老系统上无效）
    [ -k FILE ]：如果 FILE 存在，且已经设置了冒险位，则返回为真。
    [ -p FILE ]：如果 FILE 存并，且是命令管道时，返回为真。
    [ -s FILE ]：如果 FILE 存在，且大小非 0 时，则返回为真。
    [ -u FILE ]：如果 FILE 存在，且设置了 SUID 位时返回为真。
    [ -O FILE ]：如果 FILE 存在，且属有效用户 ID，则返回为真。
    [ -G FILE ]：如果 FILE 存在，且默认组为当前组，则返回为真。（只检查系统默认组）
    [ -L FILE ]：如果 FILE 存在，且是一个符号连接，则返回为真。
    [ -N FILE ]：如果 FILE 存在，且自从它最后一次读取之后，内容有被修改，则返回为真。
    [ -S FILE ]：如果 FILE 存在，且是一个套接字，则返回为真。
    ```

## 2. 字符串比较运算符

（请注意引号的使用，这是防止空格扰乱代码的好方法）

1. -z string

    如果 string 长度为零，则为真，例如：[ -z "$myvar" ]

2. -n string

    如果 string 长度非零，则为真，例如：[ -n "$myvar" ]

3. string1 == string2

    如果 string1 与 string2 相同，则为真，例如：["$myvar" == "one two three"]

4. string1 != string2

    如果 string1 与 string2 不同，则为真，例如：["$myvar" != "one two three"]

5. string1 < string2

    如果 string1 小于 string2 相同，则为真，例如：["$myvar" < "one two three"]

6. string1 > string2

    如果 string1 大于 string2 不同，则为真，例如：["$myvar" > "one two three"]

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

4. 其他

    ```bash
    [ ] || [ ] ：用 OR 来合并两个条件
    [ ] && [ ] ：用 AND 来合并两个条件
    ```

## 5. 其他

- [ -t FD ] 如果文件描述符 FD （默认值为1）打开且指向一个终端则返回为真
- [ -o optionname ] 如果shell选项optionname开启则返回为真

## 6. 查看磁盘、文件大小 

1. df -h

    查看磁盘占用情况

2. du -sm ./*

    查看当前目录下文件大小，单位 M

## 7. 参考文章

- [linux shell 指令 诸如-d, -f, -e之类的判断表达式简介](https://www.cnblogs.com/zrmw/p/9625727.html)

- [shell 中的 if 语句](https://www.cnblogs.com/kaishirenshi/p/9729800.html)
