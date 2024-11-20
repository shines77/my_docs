# QT常见报错解决方案

## 1. 常见报错

- TypeError: Property 'asciify' of object Core::Internal::UtilsJsExtension(0x41cf6f8) is not a function"

    [【Qt5.8】TypeError: Property 'asciify' of object Core问题解决办法](https://blog.51cto.com/dlican/3740664)

    - 方法一：以管理员身份运行 QT Creator；
    - 方法二：项目 -> 构建设置 -> 去掉 "Shadow build"；

- source\weatherpane.cpp(80): error C2001: 常量中有换行符

    [Qt开发，编译报错：error: C2001: 常量中有换行符](https://blog.csdn.net/weixin_43782998/article/details/132082173)

    [Qt开发，报错：error: C2001: 常量中有换行符](https://blog.csdn.net/weixin_43782998/article/details/121032388)

    问题分析：代码编码格式导致，程序中有对中文编码格式处理，而文件格式不是中文格式。

    解决方案：

    Qt菜单 -> 工具 -> 文本编辑器 -> 行为，将默认编码更改为 UTF-8，并选择 “如果编码是UTF-8则添加”。

    Qt菜单 -> 编辑 -> 选择编码 -> 文本编码 -> UTF-8 -> 按编码保存。

    解决乱码：

    ```cpp
    // 首先加上格式配置
    #include <qtextcodec.h>
    QTextCodec::setCodecForLocale(QTextCodec::codecForName("GB2312"));

    // 然后
    ui->textEdit->append(QString::fromUtf8("中文"));

    替换为：

    ui->textEdit->append(QString::fromLocal8Bit("中文"));
    ```
