# QT的常用控件和布局

## 1. 基类

所有使用 Qt 对象模型的类都必须继承自 QObject 类，包括 QWidget、QMainWindow 等。

可以获取 parent(), children() 属性，以及使用 findChild(), findChildren() 方法来实现遍历，例如：

```cpp
QObject * parent = myObject->parent();

// 获取对象的所有子对象
QList<QObject *> children = parentObject->children();

// 获取指定名称的对象的子对象
QObject * childObject = parentObject->findChild<QObject *>("childObjectName");
QList<QObject *> children = parentObject->findChildren<QObject *>("childObjectName")
```

事件传递：

```cpp
bool MyObject::event(QEvent *event) {
    if (event->type() == QEvent::KeyPress) {
        // 处理键盘事件
        return true;
    }
    // 默认事件处理方式
    return QObject::event(event);
}
```

## 2. Qt 坐标系

坐标系转换：

```cpp
QWidget::mapToGlobal(const QPoint &);   // 将窗口坐标转换为屏幕坐标。
QWidget::mapFromGlobal(const QPoint &); // 将屏幕坐标转换为窗口坐标。
```

## 3. 信号与槽机制

### 3.1 默认信号和槽

在 Qt 中，可以使用默认信号和槽的机制，它允许你不需要手动声明信号和槽的情况下，使用 Qt 提供的默认信号和槽来进行连接。默认信号和槽的机制在 Qt 的部分组件中内置，例如 QPushButton、QLineEdit 的点击事件等。

示例：

```cpp
QPushButton * button = new QPushButton("Click Me");

QObject::connect(button, &QPushButton::clicked, [=]() {
    qDebug() << "Button Clicked!";
});
```

在这个例子中，QPushButton 的 clicked 信号与 lambda 表达式连接，当按钮被点击时，lambda 表达式内的代码会执行。

### 3.2 其他组件的默认信号和槽

其他一些组件也有默认的信号和槽，例如 QLineEdit 的 textChanged 信号。

示例：

```cpp
QLineEdit * lineEdit = new QLineEdit();

QObject::connect(lineEdit, &QLineEdit::textChanged, [=](const QString &text) {
    qDebug() << "Text Changed: " << text;
});
```

### 3.3 Qt 自定义信号和槽

在 Qt 中，你可以自定义信号和槽，使得不同对象之间可以进行自定义的事件通信。

自定义信号和槽的使用步骤如下：

1. 在类的声明中声明信号(Signal)：

在类的声明中使用 `signals` 关键字声明信号。信号可以带参数，参数类型可以是任何 Qt 支持的数据类型。

```cpp
class MyObject : public QObject {
    Q_OBJECT
public:
    MyObject() {}

signals:
    void mySignal(int value);
};
```

2. 在类的声明中声明槽(Slot)：

在类的声明中使用 `slots` 关键字声明槽。槽函数可以是任何成员函数，用于处理信号触发时的操作。

```cpp
class MyObject : public QObject {
    Q_OBJECT
public:
    MyObject() {}

public slots:
    void mySlot(int value) {
        qDebug() << "Received value: " << value;
    }
};
```

3. 在类的实现中使用信号

在类的成员函数中使用 `emit` 关键字来触发信号。当槽函数与信号连接后，信号被触发时，连接的槽函数将被调用。

```cpp
class MyObject : public QObject {
    Q_OBJECT
public:
    MyObject() {}

signals:
    void mySignal(int value);

public slots:
    void triggerSignal() {
        emit mySignal(42);  // 触发信号并传递参数
    }
};
```

4. 信号(Signal)与槽(Slot)的连接

使用 QObject::connect() 函数将信号(Signal)与槽(Slot)连接起来。连接成功后，当信号被触发时，与之连接的槽函数将被调用。

```cpp
MyObject * obj = new MyObject();
QObject::connect(obj, &MyObject::mySignal, obj, &MyObject::mySlot);

// 当信号触发时, mySlot() 函数将被调用
obj->triggerSignal();
```

这种自定义信号(Signal)和槽(Slot)的机制使得不同对象能够在特定事件发生时进行通信，是 Qt 中事件驱动编程的核心机制之一。

## 4. lambda 表达式

Lambda 表达式在 Qt 中非常常用，特别是在信号与槽的连接、算法和 STL 容器的使用等方面。

1. Lambda 表达式基本语法：

    - \[捕获列表\](参数列表) -> 返回类型 { 函数体 }

2. 捕获列表：

    - **[=]**：以传值方式捕获所有局部变量。
    - **[&]**：以引用方式捕获所有局部变量。
    - **[变量]**：捕获特定变量，可以使用 = 或 & 指定捕获方式。

3. Lambda 表达式作为匿名函数：

    - 可以在需要函数的地方使用 Lambda 表达式，不需要显示地声明函数。

4. Lambda 表达式的一些常用用法：

    ```cpp
    // 用于连接信号与槽
    QObject::connect(sender, &SenderClass::signalName, [=]() {
        // Lambda表达式内的代码
    });

    // 用于 STL 算法
    std::vector<int> numbers = { 1, 2, 3, 4, 5 };
    std::for_each(numbers.begin(), numbers.end(), [](int number) {
        qDebug() << number;
    });

    // 用于 STL 容器
    std::vector<int> numbers = { 5, 2, 8, 1, 3 };
    std::sort(numbers.begin(), numbers.end(), [](int a, int b) {
        return a < b;
    });

    // 用于 Qt 的一些算法
    QVector<int> numbers = { 1, 2, 3, 4, 5 };
    QVector<int> result = QtConcurrent::blockingMapped(numbers, [](int number) {
        return number * 2;
    });
    ```

## 5. QMainWindow

QMainWindow 是 Qt 框架中的一个核心类，用于创建应用程序的主窗口。它提供了一个标准的应用程序主窗口结构，包括菜单栏、工具栏、状态栏等，并且可以容纳各种自定义的用户界面组件。

### 5.1 主要特性

- **菜单栏（Menu Bar）**：QMenuBar 对象用于创建菜单栏，菜单栏通常包含一个或多个菜单，每个菜单包含若干个菜单项。

- **工具栏（Tool Bar）**：QToolBar 对象用于创建工具栏，工具栏通常包含一组快捷操作按钮，用于执行常用的功能。

- **状态栏（Status Bar）**：QStatusBar 对象用于创建状态栏，用于显示应用程序的状态信息、提示信息等。

- **中央部件（Central Widget）**：通常是一个自定义的QWidget派生类，作为主窗口的中央区域，用于放置应用程序的主要内容。

- **Dock窗口（Dock Widgets）**：QDockWidget 对象用于创建可停靠的面板，用户可以拖动和停靠这些面板。

- **文档模型/视图架构（Document-View Architecture）**：QMainWindow 支持文档-视图架构，可以通过多文档界面（MDI）或单文档界面（SDI）的方式管理多个文档。

- **布局管理（Layout Management）**：QMainWindow 支持布局管理，可以方便地安排各种子组件的位置和大小。

### 5.2 常用方法和函数

- **setMenuBar(QMenuBar * menuBar)**：设置菜单栏。
- **addToolBar(QToolBar * toolbar)**：添加工具栏。
- **setStatusBar(QStatusBar * statusBar)**：设置状态栏。
- **setCentralWidget(QWidget * widget)**：设置中央部件。
- **addDockWidget(Qt::DockWidgetArea area, QDockWidget * dockwidget)**：添加 Dock 窗口。

## x. 参考文章

- [QT入门看这一篇就够了](https://blog.csdn.net/arv002/article/details/133785137)
