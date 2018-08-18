
# Golang Http Router

* [`httprouter框架 (Gin使用的路由框架)`](http://www.okyes.me/2016/05/08/httprouter.html) by Tao Peng(Vector)

    [http://www.okyes.me/2016/05/08/httprouter.html](http://www.okyes.me/2016/05/08/httprouter.html)

    
    ```
    在 Gin 中已经说到, Gin 比 Martini 的效率高好多耶, 究其原因是因为使用了 httprouter 这个路由框架, httprouter 的 git 地址是:  [httprouter 源码](https://github.com/julienschmidt/httprouter). 今天稍微看了下 httprouter 的 现原理, 其实就是使用了一个 radix tree (前缀树) 来管理请求的 URL , 下面介绍一下 httprouter 原理.

    主要介绍了：

    1. httprouter 的基本结构;
    2. 建树的过程;
    3. 查找 path 的过程;

    注：代码中包含部分中文注释。
    ```

* [`Go HTTP路由调研`](http://www.csyangchen.com/go-http-router.html) by csyangchen

    [http://www.csyangchen.com/go-http-router.html](http://www.csyangchen.com/go-http-router.html)

    ```
    介绍和对比了：

    1. http.ServeMux
    2. httprouter
    3. gin
    4. beego

    1. http.ServeMux

    http.ServeMux 是标准库自带的 URL 路由. 其实现比较简单, 每个路径注册到一个字典里面, 查找的时候, 遍历字典, 并匹配最长路径.

    2. httprouter

    目前项目使用的是 gin, 其路由采用了 httprouter. 相对于 http.ServeMux, httprouter支持:

        1. 根据方法注册路由
        2. 支持动态路由

    3. gin

    gin 的路由器是基于 httprouter 的. 提几个比较有用的功能:

        1. 链式, 插件化的中间层(MiddleWare)模块支持, 很方便增加日志/监控/限流等通用功能
        2. 支持路由组(RouterGroup)的写法, 从而注册HTTP服务模块不需要关心服务的绝对路径, 方便组合
        3. 数据绑定(Binding)等帮助方法, 当然这些和我们这里主要讨论的路由功能就扯得比较远了

    4. BeeGo

    beego 是国人开发的Web开发框架, 在 go-http-routing-benchmark 中, 其路由性能似乎表现不佳, 我们来深究下原因.

    看一下其路由实现, 也是使用了查找树, 但是对子节点查找时, 需要遍历, 而 httprouter 的每个节点, 保存了对于其子节点的路由信息 node.indices, 查找上自然更快. 此外, beego 路由查找方法使用了递归的方式 (Tree.match), 而 httprouter 在一个执行循环 (node.getValue) 里就可以搞定, 自然效率更高.
    ```

* [`[julienschmidt]: Go HTTP Router Benchmark`]((https://github.com/julienschmidt/go-http-routing-benchmark))

    [GitHub] : [https://github.com/julienschmidt/go-http-routing-benchmark](https://github.com/julienschmidt/go-http-routing-benchmark)

    ```
    This benchmark suite aims to compare the performance of HTTP request routers for Go by implementing the routing structure of some real world APIs. Some of the APIs are slightly adapted, since they can not be implemented 1:1 in some of the routers.

    Of course the tested routers can be used for any kind of HTTP request → handler function routing, not only (REST) APIs.
    Tested routers & frameworks:

        01. Beego
        02. go-json-rest
        03. Denco
        04. Gocraft Web
        05. Goji
        06. Gorilla Mux
        07. http.ServeMux
        08. HttpRouter
        09. HttpTreeMux
        10. Kocha-urlrouter
        11. Martini
        12. Pat
        13. Possum
        14. R2router
        15. TigerTonic
        16. Traffic
    ```

* [`https://github.com/julienschmidt/httprouter`](https://github.com/julienschmidt/httprouter)

    [GitHub] : [https://github.com/julienschmidt/httprouter](https://github.com/julienschmidt/httprouter)

    ```
    HttpRouter is a lightweight high performance HTTP request router (also called multiplexer or just mux for short) for Go.

    In contrast to the default mux of Go's net/http package, this router supports variables in the routing pattern and matches against the request method. It also scales better.

    The router is optimized for high performance and a small memory footprint. It scales well even with very long paths and a large number of routes. A compressing dynamic trie (radix tree) structure is used for efficient matching.
    ```

*  [`https://godoc.org/github.com/julienschmidt/httprouter`](https://godoc.org/github.com/julienschmidt/httprouter)

    [GoDoc] : [https://godoc.org/github.com/julienschmidt/httprouter](https://godoc.org/github.com/julienschmidt/httprouter)

    # package httprouter #

    ```golang
    import "github.com/julienschmidt/httprouter"
    ```

    Package `httprouter` is a trie based high performance `HTTP` request router.

    A trivial example is:

    ```golang
    package main

    import (
        "fmt"
        "github.com/julienschmidt/httprouter"
        "net/http"
        "log"
    )

    func Index(w http.ResponseWriter, r *http.Request, _ httprouter.Params) {
        fmt.Fprint(w, "Welcome!\n")
    }

    func Hello(w http.ResponseWriter, r *http.Request, ps httprouter.Params) {
        fmt.Fprintf(w, "hello, %s!\n", ps.ByName("name"))
    }

    func main() {
        router := httprouter.New()
        router.GET("/", Index)
        router.GET("/hello/:name", Hello)

        log.Fatal(http.ListenAndServe(":8080", router))
    }
    ```

* [`Golang：使用 httprouter 构建 API 服务器`](http://oopsguy.com/2017/10/20/build-api-server-with-httprouter-in-golang/)

    [http://oopsguy.com/2017/10/20/build-api-server-with-httprouter-in-golang/](http://oopsguy.com/2017/10/20/build-api-server-with-httprouter-in-golang/)

    ```
    https://medium.com/@gauravsingharoy/build-your-first-api-server-with-httprouter-in-golang-732b7b01f6ab

    原作者：Gaurav Singha Roy
    译  者：oopsguy.com

    我 10 个月前开始成为一名 Gopher，没有回头。像许多其他 gopher 一样，我很快发现简单的语言特性对于快速构建快速、可扩展的软件非常有用。当我刚开始学习 Go 时，我正在玩不同的多路复用器（multiplexer），它可以作为 API 服务器使用。如果您像我一样有 Rails 背景，你可能也会在构建 Web 框架提供的所有功能方面遇到困难。回到多路复用器，我发现了 3 个是非常有用的好东西，即 Gorilla mux、httprouter 和 bone（按性能从低到高排列）。即使 bone 有最佳性能和更简单的 handler 签名，但对于我来说，它仍然不够成熟，无法用于生产环境。因此，我最终使用了 httprouter。在本教程中，我将使用 httprouter 构建一个简单的 REST API 服务器。
    ```