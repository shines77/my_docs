# 消除 Unused variable 的几种方式

## 1. 消除 Unused variable

### 1.1 __attribute__()

```cpp
type varname __attribute__((unused));
or
type varname __unused;
```

### 1.2 `#pragma unused()`

```cpp
type varname = ...;
#pragma unused(varname)
```

### 1.3 最原始的方法

简单有效：

```cpp
type varname = ...;
(void)(varname);
```

## 2. 参考文章

1. `[如何屏蔽Unused variable]`

    [https://www.jianshu.com/p/5618898aa7a0](https://www.jianshu.com/p/5618898aa7a0)
