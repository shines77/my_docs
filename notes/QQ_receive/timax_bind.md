# 使用timax::bind， 为你获取一个std::function对象
## Motivation
刚刚发布的[rest_rpc](https://github.com/topcpporg/rest_rpc)，接口易用性很高。例如服务器注册业务函数的接口：
```cpp
server.register_handler("add", client::add);
```
我们希望将字符串"add"与业务函数client::add绑定。在示例中add是一个类型为int(int,int)的普通函数。除了普通函数，我们希望这个接口可以接受任何[callable](http://en.cppreference.com/w/cpp/concept/Callable)类型的对象，包括普通函数、仿函数、lambda表达式和任何重载了一个非泛型的operator()的类型对象，这个目的可以通过function_traits，可以很容易的办到。

但是如何兼容类的非静态成员函数？
```cpp
class foo                                           // 假设我们有一个类foo
{
    int add(int, int);                              // 它有一个名为add类型为int(int,int)的非静态成员函数
}

foo f;                                              // 假设我们还有一个foo的对象实例

server.register_handler("add", &client::add, f);    // 我们还需要一个像这样的接口
```
使用[std::bind](http://en.cppreference.com/w/cpp/utility/functional/bind)？很遗憾，C++标准并没有明确定std:bind的返回值。更遗憾的是，std::bind返回的类型是一个拥有泛型operator()的结构体，也就是说，它的operator()是一个模板函数。因此，我们无法用function_traits做进一步的处理。而且适用std::bind，还有一个不便利的地方，我们在使用std::bind的时候，还需要填补placehodler.
```cpp
using namespace std::placehodlers;
server.register_handler("add", std::bind(&client::add, f, _1, _2));
```
于是，我们决定撸一个[timax::bind](https://github.com/topcpporg/rest_rpc/blob/master/rest_rpc/base/function_traits.hpp)，直接为用户返回std::function<Ret(Args...)>对象，这样可以让funtion_traits正常工作。

## Design
### 什么该做，什么不该做
timax::bind并没有重新实现一个std::bind，而是封装并扩展了std::bind. 重复造轮子的工作不该做，而让std::bind在我们的应用场景下更好用是应该做的事情。timax::bind扩展的工作主要有两项：
1. 包装std::bind，为用户返回一个std::function对象；
2. 在函数参数都默认的情况下，不需要用户填上placehodler；
### 要做的工作
timax::bind的设计思路如下图：（橙色部分是需要完成实现的工作）

![](http://purecpp.org/wp-content/uploads/2016/09/timax_bind1.png)

我们用代码来详细解释一下
```cpp
#include <rest_rpc/rest_rpc.hpp>

namespace client
{
    int add(int a, int b) { return a + b; }

    struct foo
    {
        int add(int a, int b) { return a + b; }
    };
}

int main()
{
    // 用timax::bind绑定普通callable
    {
        auto bind_1 = timax::bind(client::add, 1 ,2);                   // 全部都是实参
        auto bind_2 = timax::bind(client::add, _1, 2);                  // 部分是实参，部分是占位符
        auto bind_3 = timax::bind(client::add, _2, _1);                 // 全部都是占位符
        auto bind_4 = timax::bind(client::add);                         // 没有绑定任何参数
    }

    // 用timax::bind绑定类的非静态成员函数
    {
        foo f;                                                          // 普通对象
        auto f_ptr = std::make_shared<foo>();                           // 智能指针
        auto f_raw = new foo;                                           // 裸指针

        auto bind_1 = timax::bind(&client::foo::add, f, 1, 2);          // bind类对象，with实参
        auto bind_2 = timax::bind(&client::foo::add, f_ptr, 2, 3);      // bind智能指针，with实参
        auto bind_3 = timax::bind(&client::foo::add, f_raw, 3, 5);      // bind裸指针，with实参
        auto bind_4 = timax::bind(&client::foo::add, f_ptr, _2, 3);     // bind智能指针，with部分实参与部分占位符
        auto bind_5 = timax::bind(&client::foo::add, f_raw, _1, _2);    // bind裸指针，with占位符
        auto bind_6 = timax::bind(&client::foo::add, f);                // bind对象，without参数
        
        delete f_raw;
    }
    return 0;
}
```

## Implementation
timax::bind的实现，由上图可以看出，主要工作是两个方面。
1. 推导std::function<Ret(Args...)>中的Ret和Args...；
2. 返回一个可以构造std::function<Ret(Args...)>的对象；

推导参数主要看timax::bind是否绑定了参数，或者占位符。因为，绑定了额外的参数，会让函数的形参数目和顺序发生改变。在不绑定任何参数的情况下，是最简单的，同原来函数的返回类型与形参列表相同。
### 无额外绑定参数情形
正如上文所述，这种情况非常简单，因为返回类型很好确定。我们可以使用function_traits，推导出返回的std::function模板实例化后的类型。那么剩下的工作就是要构造一个对象，能够正确调用我们绑定的函数，并初始化std::function实例化后的对象实例。根据Design章节的附图，我们分成四种情况：
1. 绑定的是一个callable（函数、仿函数、lambda表达式和函数对象）；
2. 用一个类对象绑定一个类的非静态成员变量；
3. 用一个类的裸指针绑定一个类的非静态成员变量；
4. 用一个类的智能指针绑定一个类的非静态成员变量；

绑定一个普通callable的实现很简单，我们用一个lambda表达式包装了一下调用。这里用到了C++14的auto lambda表达式，使用auto和variadic特性，让编译器自动帮我们填写函数的signature. 如果没有C++14的特性，还真不知道怎么实现比较方便，欢迎各路大神在这里指教一下，如果只用C++11的特性来实现。
```cpp
template <typename F>
auto bind_a_normal_callable(F&& f) -> typename function_traits<F>::stl_function_type
{
    return [func = std::forward<F>(f)](auto&& ... args){ return func(std::forward<decltype(args)>(args)...); };
}
```
剩下三种情况的实现方法，也很明了了，唯一要确定的是调用语义。而且，我们的实现是<b>支持多态</b>的。
```cpp
struct caller_is_a_pointer {};                          // 调用者是一个裸指针
struct caller_is_a_smart_pointer {};                    // 调用者是一个智能指针
struct caller_is_a_lvalue {};                           // 调用者是一个对象

template <typename F, typename Caller>
auto bind_impl_pmf(caller_is_a_pointer, F&& pmf, Caller&& caller)
    -> typename function_traits<F>::stl_function_type
{
    // 使用指针的operator ->来访问pmf
    return [pmf, c = std::forward<Caller>(caller)](auto&& ... args) { return (c->*pmf)(std::forward<decltype(args)>(args)...); };	
}

template <typename F, typename Caller>
auto bind_impl_pmf(caller_is_a_smart_pointer, F&& pmf, Caller&& caller)
    -> typename function_traits<F>::stl_function_type
{
    // 智能指针不能像裸指针一样直接访问pmf，要先get再使用裸指针的语义
    return[pmf, c = std::forward<Caller>(caller)](auto&& ... args) { return (c.get()->*pmf)(std::forward<decltype(args)>(args)...); };
}

template <typename F, typename Caller>
auto bind_impl_pmf(caller_is_a_reference, F&& pmf, Caller&& caller)
    -> typename function_traits<F>::stl_function_type
{
    // 对象直接使用operator .来访问pmf
    return [pmf, c = std::forward<Caller>(caller)](auto&& ... args) { return (c.*pmf)(std::forward<decltype(args)>(args)...); };
}
```
最后利用模板元，特化和重载决议等一些modern C++的惯用法，把这些实现像用胶水一样粘起来，统一成一个接口给外层用就可以了。

### 有额外参数绑定的情形
有额外参数的情形就麻烦一些了，因为函数的形参列表发生了改变。绑定了实参，形参列表会变小；绑定了占位符，形参列表的顺序可能会改变。那么如何计算返回的std::function的实例化类型呢？切入点在placeholder!
```cpp
// 假设我们有一个signature为double(int, std::string const&, float)的函数
double foo(int, std::string const&, float);

timax::bind(foo, 1, _2, _1);            // signature为int(flaot, std::string const&)
timax::bind(foo, _3, _1, _2);           // signature为int(float, int, std::string const&)
timax::bind(foo, 1, "timax", 0.0f);     // signature为int();
```
也就是说，我们在计算返回类型与形参列表的类型的时候，<b>只需要考虑placeholder而可以忽略所有实参</b>。接下来继续用代码来描述计算类型额过程。
```cpp
// 依旧假设我们有一个signature为double(int, std::string const&, float)的函数
double foo(int, std::string const&, float);

// 对于这个函数，它的形参列表变形为tuple的类型是
using args_tuple_t = std::tuple<int, std::string const&, float>;

// 假设我们有一个bind的调用
timax::bind(foo, _3, _1, _2);

// 那么我们把绑定的占位符所代表的编译期常量放入std::integer_sequence<int, ...>
using bind_ph_seq_t = std::integer_sequence<int, 3, 1, 2>;

// 那么得到的std::function实例化的类型就是
using function_t = std::function<int(
    std::tuple_element_t<3, args_tuple_t>,
    std::tuple_element_t<1, args_tuple_t>,
    std::tuple_element_t<2, args_tuple_t>
)>;

// 那么也就是std::function<float, std::string const&, int>
static_assert(std::is_same<function_t, std::function<float, std::string const&, int>>::value, "");
```
对于有绑定实参的情形，可以看出，我们只关心占位符placeholder，所以结果也完全正确。如果没有任何占位符，那么形参列表就是void. 在rest_rpc中，timax::bind_to_function模板就是负责推导std::function实例化类型的元函数，详细可以参见[function_traits.hpp](https://github.com/topcpporg/rest_rpc/blob/master/rest_rpc/base/function_traits.hpp)的151行。
剩下的事情就是调用std::bind了，这就很简单了。
```cpp
template <typename F, typename Arg0, typename ... Args>
auto bind_impl_callable(F&& f, Arg0&& arg, Args&& ... args)
    -> typename bind_to_function<F, Args...>::type
{
    // 就是计算一下返回类型，再调用std::bind的一层壳
    return std::bind(std::forward<F>(f), std::forward<Arg0>(arg0), std::forward<Args>(args)...);
}

template <typename F, typename Caller, typename Arg0, typename ... Args>
auto bind_impl_pmf(F&& pmf, Caller&& caller, Arg0&& arg0, Args&& ... args)
    -> typename bind_to_function<typename function_traits<F>::function_type, Arg0, Args...>::type
{
    // 只是一层壳，额外工作就是计算了返回类型
    return std::bind(pmf, std::forward<Caller>(caller), std::forward<Arg0>(arg0), std::forward<Args>(args)...);
}
```
## Tail
timax::bind的实现特别简单，但是为rest_rpc注册业务函数带来了便利。这就是C++的魅力，在语言工具和标准库功能不够的时候，你随时都有可能为自己定制和扩展C++. 定制和扩展势必会用到一些模板元和宏元，会写一些思路不同于常规程序的代码。但这些代码的出现，是为了让常规代码更符合人类直观感受的逻辑思维。看到一些人喜欢妖魔化C++的种种特性，甚至把<b>回字的四种写法</b>拿出来嘲讽这些代码是咬文嚼字，工笔对偶的华而不实。我只能说，不同的需求关心的问题大相径庭。在编写库的时候，需要实现者反复琢磨，接口是否符合常规逻辑思维，是否类型安全，是否是富类型能涵盖所有等等，才有了使用者顺滑的开发体验。《C++语言的设计与演化》诠释了BS对C++设计哲学的源点，不以个人喜好来增删语言的特性，所以我们才能看到一个强大的编程语言。~~然而新标准也出的忒慢了吧！~~ 愿rest_rpc功能越来越完善，C++er越来越好，也欢迎各路大神来访purecpp.org和QQ群296561497一起讨论C++的开(~~奇~~)发(~~淫~~)经(~~巧~~)验(~~记~~)。