# CMake 技巧和经验

## 1. 使用经验

* [CMAKE 里 PRIVATE、PUBLIC、INTERFACE 属性示例详解](https://blog.csdn.net/weixin_43862847/article/details/119762230)

  * PRIVATE: 只对内公开，对外不公开。
  * INTERFACE：只对外公开，对内不公开。
  * PUBLIC：对内，对外都公开。

* [CMake 中的 PUBLIC，PRIVATE，INTERFACE](https://www.jianshu.com/p/07761ff7838e)

    从使用的角度解释，若有 C 链接了目标 A：

  * 如果依赖项 B 仅用于目标 A 的实现，且不在头文件中提供给C 使用，使用 PRIVATE。
  * 如果依赖项 B 不用于目标 A 的实现，仅在头文件中作为接口提供给 C 使用，使用 INTERFACE。
  * 如果依赖项 B 不仅用于目标 A 的实现，而且在头文件提供给 C 使用，使用 PUBLIC。
