
配置 EditPlus 支持 Shell 语法高亮和自动补全
===============================================

# 1. 文件下载 #

我们可以从 `EditPlus` 官网的 “`User Files`” 获得不同语言的语法高亮和自动补全文件。

`Bash` 版本下载地址为：[http://www.editplus.com/dn.php?n=bash.zip](http://www.editplus.com/dn.php?n=bash.zip)

# 2. 配置EditPlus支持Shell语法高亮和自动补全 #

1) 解压文件，并放到 “`C:\Users\[用户名]\AppData\Roaming\EditPlus 3`” 下（`Win7` 默认位置）。当然，也可以放到其他位置。

2) `Tools` -> `Preferences` -> `Files` -> `Settings & syntax`

    点击 "`Add...`" 按钮添加一个新的文件类型。在这里，填入 “`Bash`”。

3) `File Extensions` 填写 “`sh`”；

4) `Syntax Files` 选择语法高亮文件（`bash.stx`）；

5) `Auto Completion` 选择自动补全文件（`bash.acp`）；

6) `Tab/Intent`：将 `Tab` 和 `Intent` 都改为 `4`，勾选 “`Insert spaces instead of tab`”（使用空格代替 `Tab`）；

7) `Function Pattern` 里填写：`function[ \t]+[0-9a-zA-Z_]+[ \t]*\([ \t]*\)`。这样就可以使用 “`Ctrl + F11`” 查看函数列表了。

# 3. 参考文章 #

1. [配置EditPlus支持Shell语法高亮和自动补全](http://www.pythoner.com/182.html)

<.End.>
