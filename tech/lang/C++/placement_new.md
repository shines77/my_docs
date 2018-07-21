
# Placement New 的用法

* `理解 C++ placement new 语法`

    [https://www.jianshu.com/p/4af119c44086]()

    ```cpp
    new 表达式的语法如下:

    new new-type-id ( optional-initializer-expression-list )

    new ( expression-list ) new-type-id ( optional-initializer-expression-list )

    void * operator new (std::size_t) throw(std::bad_alloc);
    void operator delete (void *) throw();

    什么是 placement operator new(), 它首先是一个函数, 而且是标准 operator new() 的一个重载函数. wiki 中这么定义它:
    // The "placement" versions of the new and delete operators and functions are known as placement new and placement delete.

    // 标准版本
    void * operator new (std::size_t) throw(std::bad_alloc);
    // nothrow 版本
    void * operator new (std::size_t, const std::nothrow_t &) throw();
    // 标准版本
    void operator delete (void * pMemory) throw();
    // nothrow 版本
    void operator delete (void * pMemory, const std::nothrow_t & nt) throw();

    // default placement new
    void * operator new (std::size_t, void * ptr) throw() { return p ; }
    void operator delete (void * pMemory, void * ptr) throw() { }
    ```

* `[Effective C++] Item 52: 如果编写了 placement new，就要编写 placement delete`

    [https://wizardforcel.gitbooks.io/effective-cpp/content/54.html](https://wizardforcel.gitbooks.io/effective-cpp/content/54.html)

    ```cpp
    Effective C++
    
    作者：Scott Meyers
    译者：fatalerror99 (iTePub's Nirvana)
    发布：http://blog.csdn.net/fatalerror99/

    #include <new>

    // normal new
    void * operator new(std::size_t size) throw(std::bad_alloc);

    // placement new
    void * operator new(std::size_t size, void * rawMemory) throw();

    // nothrow new
    void * operator new(std::size_t size,
                        const std::nothrow_t & nothrow) throw();

    // normal delete
    void operator delete(void * rawMemory) throw();    // normal signature
                                                       // at global scope

    // placement delete
    void operator delete(void * rawMemory,             // typical normal
                         std::size_t size) throw();    // signature at class
                                                       // scope
    // nothrow delete
    void operator delete(void * rawMemory,                  // typical normal
                         const std::nothrow_t &) throw();   // signature at class
                                                            // scope

    class StandardNewDeleteForms {
    public:
        // normal new/delete
        static void * operator new(std::size_t size) throw(std::bad_alloc) {
            return ::operator new(size);
        }

        static void operator delete(void * pMemory) throw() {
            ::operator delete(pMemory);
        }

        // placement new/delete
        static void * operator new(std::size_t size, void * ptr) throw() {
            return ::operator new(size, ptr);
        }

        static void operator delete(void * pMemory, void * ptr) throw() {
            return ::operator delete(pMemory, ptr);
        }

        // nothrow new/delete
        static void * operator new(std::size_t size, const std::nothrow_t & nt) throw() {
            return ::operator new(size, nt);
        }

        static void operator delete(void * pMemory, const std::nothrow_t &) throw() {
            ::operator delete(pMemory);
        }
    };
    ```