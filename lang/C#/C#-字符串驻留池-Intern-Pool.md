
# Java、C# 字符串常量池（Intern Pool）

* `原来是这样：C# 中字符串的内存分配与驻留池`

    [https://www.cnblogs.com/instance/archive/2011/05/24/2056091.html](https://www.cnblogs.com/instance/archive/2011/05/24/2056091.html)

    ```csharp
    基于 String 类这样的特性，CLR 让表示相同的字符串实际值的变量指向同一个 String 事例。

    因为利用任何一个对 String 实例的引用所进行的修改操作都不会切实地影响到该实例的状态，也就不会影响到其他所有指向该实例的引用所表示的字符串实际值。CLR 如此管理 String 类的内存分配，可以优化内存的使用情况，避免内存中包含冗余的数据。

    为了实现这个机制，CLR 默默地维护了一个叫做驻留池（Intern Pool）的表。这个表记录了所有在代码中使用字面量声明的字符串实例的引用。这说明使用字面量声明的字符串会进入驻留池，而其他方式声明的字符串并不会进入，也就不会自动享受到 CLR 防止字符串冗余的机制的好处了。这就是我上文提到的“某些情况下，确实也会发生同一个字符串实际值在内存中有多份副本同时存在”的例子。
    ```

    ```csharp
    StringBuilder sb = new StringBuilder();
    sb.Append("He").Append("llo");

    string s1 = "Hello";
    string s2 = String.Intern(sb.ToString());

    bool same = (object) s1 == (object) s2;

    var s3 = "hello";
    var s4 = "hello";

    Console.WriteLine(object.ReferenceEquals(s3, s4));
    ```
