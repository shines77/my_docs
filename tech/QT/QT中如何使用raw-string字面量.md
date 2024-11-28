# QT 中如何使用 raw-string 字面量

## 1. 需求

在 Qt 中，有时候需要给一个 Widget 设置 qss，但 qss 内容比较长，还包含很多换行，我们想直接从 css 文件直接把字符串复制过来，怎么办？

这个时候可以使用 C++ 11 引入的 raw string literals (原始字符串字面量)。

## 2. raw string literals

raw string literals (原始字符串字面量) 的语法如下：

```cpp
R"delimiter(string)delimiter"
```

- `delimiter` 是可选的分隔符，帮助标识字符串的开始和结束。它是一个可选的自定义标签，可以省略。如果省略，默认使用括号 " 和 "。
- `string` 是你要表示的字符串内容，它会被原样保留。

## 3. 用法

要在 Qt 中使用 raw string 字面量，编译器必须支持 C++ 11，Visual Studio 需要 2015 以及 2015 以后的版本。

同时，要在 qmake 项目配置文件 xxxxxxxx.pro 中添加如下设置：

```bash
CONFIG += c++11
```

或者在 cmake 配置文件 CMakeLists.txt 中加入 C++ 11 的编译选项，例如：

```bash
set(CMAKE_CXX_FLAGS_DEFAULT "${CMAKE_CXX_FLAGS} -std=c++11 -march=native -mtune=native -Wall -fPIC")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_DEFAULT} -O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS_DEFAULT} -g -pg -D_DEBUG")
```

### 3.1 表示文件路径

在 Windows 下，路径分隔符 `\` 和 字符串的转义符 `\` 冲突，所以表示 Windows 路径会用 `\\` 来表示一个 `\` 字符，使用 raw string 字面量则更方便一点。

例如：

```cpp
int main()
{
    std::string str1 = "D:\\hello\\world\\test.text";
    std::cout << str1 << std::endl;

    std::string str2 = R"(D:\hello\world\test.text)";  // raw string 字面量
    std::cout << str2 << std::endl;

    return 0;
}
```

### 3.2 更长的字符串

需要注意的是，使用了 raw string 字面量，字符串中的 `\n` 将不再表示换行，而是直接原样输出 `\n` 字符。

例如：

```cpp
int main()
{
    // 普通字符串
    const char * normal_string = "This is a line break:\nNew line here.\nAnd a tab:\tTab character here.";
    std::cout << "Normal String:\n\n" << normal_string << std::endl;

    // 原始字符串
    const char * raw_string = R"(This is a line break:\nNew line here.\nAnd a tab:\tTab character here.)";
    std::cout << "Raw String:\n\n" << raw_string << std::endl;

    return 0;
}
```

输出的结果为：

```
Normal String:

This is a line break:
New line here.
And a tab:    Tab character here.

Raw String:

This is a line break:\nNew line here.\nAnd a tab:\tTab character here.
```

**更多长字符串的用法**

表示 json 字符串，qss 长字符串，佛像长字符串等等。

范例：

```cpp
int main()
{
    // json 字符串
    std::string json_body = R"({
        "name": "xas",
        "age": 18,
        "sex": "男",
        "scores": [88, 77.5, 66]
 })";

    std::cout << json_body << std::endl;

    // css 长字符串
    std::string qss_style = R"(QWidget QListWidget {
    color: rgb(51, 51, 51);
    background-color: rgb(247, 247, 247);
    border: 1px solid rgb(247, 247, 247);
    outline: 0px;
}

QWidget QListWidget::Item {
    background-color: rgb(247, 247, 247);
}

QWidget QListWidget::Item:hover {
    background-color: rgb(247, 247, 247);
}

QWidget QListWidget::item:selected {
    color: black;
    background-color: rgb(247, 247, 247);
    border: 1px solid rgb(247, 247, 247);
}

QWidget QListWidget::item:selected:!active {
    border: 1px solid rgb(247, 247, 247);
    background-color: rgb(247, 247, 247);
    color: rgb(51,51,51);
})";

    std::cout << qss_style << std::endl;

    // '佛祖' 长字符串
    std::string fozu = R"(
                            _ooOoo_
                           o8888888o
                           88" . "88
                           (| -_- |)
                            O\ = /O
                        ____/`---'\____
                      .   ' \\| |// `.
                       / \\||| : |||// \
                     / _||||| -:- |||||- \
                       | | \\\ - /// | |
                     | \_| ''\---/'' | |
                      \ .-\__ `-` ___/-. /
                   ___`. .' /--.--\ `. . __
                ."" '< `.___\_<|>_/___.' >'"".
               | | : `- \`.;`\ _ /`;.`/ - ` : | |
                 \ \ `-. \_ __\ /__ _/ .-` / /
         ======`-.____`-.___\_____/___.-`____.-'======
                            `=---='
        )";

    std::cout << fozu << std::endl;

    return 0;
}

```

## 4. 参考文章

- [【C++11】详解--原始字符串字面量（多维度解析，小白一看就懂！！）](https://blog.csdn.net/weixin_45031801/article/details/140585268)
