
//
// https://en.cppreference.com/w/cpp/types/is_move_constructible
// C++ Web IDE: https://www.tutorialspoint.com/compile_cpp_online.php
//

#include <iostream>
#include <string>
#include <type_traits>
 
struct Ex1 {
    std::string str; // member has a non-trivial but non-throwing move ctor
};

struct Ex2 {
    int n;
    Ex2(Ex2&&) = default; // trivial and non-throwing
};

struct Ex3 {
    int n;
    Ex3(const Ex3 & other)  {
        n = other.n;
    }
    // non-trivial and non-throwing
    Ex3(Ex3 && other) noexcept  {
        n = other.n;
        other.n = 0;
    }
};

struct NoMove {
    // prevents implicit declaration of default move constructor
    // however, the class is still move-constructible because its
    // copy constructor can bind to an rvalue argument
    NoMove(const NoMove &) {}
};

struct NoMove_noexcept {
    // prevents implicit declaration of default move constructor
    // however, the class is still move-constructible because its
    // copy constructor can bind to an rvalue argument
    NoMove_noexcept(const NoMove_noexcept &) noexcept {}
};

struct OnlyMove {
    OnlyMove(OnlyMove &&) {}
};

struct OnlyMove_noexcept {
    OnlyMove_noexcept(OnlyMove_noexcept &&) noexcept {}
};

struct HasMove {
    HasMove(const HasMove &) {}
    HasMove(HasMove &&) {}
};


struct HasMove_noexcept {
    HasMove_noexcept(const HasMove_noexcept &) {}
    HasMove_noexcept(HasMove_noexcept &&) noexcept {}
};

struct HasMove_both_noexcept {
    HasMove_both_noexcept(const HasMove_both_noexcept &) noexcept {}
    HasMove_both_noexcept(HasMove_both_noexcept &&) noexcept {}
};

template <typename T>
void display_copy_constructible(const std::string & var)
{
    std::cout << std::boolalpha;
    std::cout << var << ":" << '\n';
    std::cout << "is copy-constructible: ";
    std::cout << std::is_copy_constructible<T>::value << '\n';
    std::cout << "is trivially copy-constructible: ";
    std::cout << std::is_trivially_copy_constructible<T>::value << '\n';
    std::cout << "is nothrow copy-constructible: ";
    std::cout << std::is_nothrow_copy_constructible<T>::value << "\n\n";
}

template <typename T>
void display_move_constructible(const std::string & var)
{
    std::cout << std::boolalpha;
    std::cout << var << ":" << '\n';
    std::cout << "is move-constructible: ";
    std::cout << std::is_move_constructible<T>::value << '\n';
    std::cout << "is trivially move-constructible: ";
    std::cout << std::is_trivially_move_constructible<T>::value << '\n';
    std::cout << "is nothrow move-constructible: ";
    std::cout << std::is_nothrow_move_constructible<T>::value << "\n\n";
}

template <typename T>
void display_copy_assignable(const std::string & var)
{
    std::cout << std::boolalpha;
    std::cout << var << ":" << '\n';
    std::cout << "is copy-assignable: ";
    std::cout << std::is_copy_constructible<T>::value << '\n';
    std::cout << "is trivially copy-assignable: ";
    std::cout << std::is_trivially_copy_constructible<T>::value << '\n';
    std::cout << "is nothrow copy-assignable: ";
    std::cout << std::is_nothrow_copy_constructible<T>::value << "\n\n";
}

template <typename T>
void display_move_assignable(const std::string & var)
{
    std::cout << std::boolalpha;
    std::cout << var << ":" << '\n';
    std::cout << "is move-assignable: ";
    std::cout << std::is_move_assignable<T>::value << '\n';
    std::cout << "is trivially move-assignable: ";
    std::cout << std::is_trivially_move_assignable<T>::value << '\n';
    std::cout << "is nothrow move-assignable: ";
    std::cout << std::is_nothrow_move_assignable<T>::value << "\n\n";
}

template <typename T>
void display_copy_and_move_all(const std::string & var)
{
    std::cout << std::boolalpha;
    std::cout << var << ":" << "\n\n";

    std::cout << "is copy-constructible: ";
    std::cout << std::is_copy_constructible<T>::value << '\n';
    std::cout << "is trivially copy-constructible: ";
    std::cout << std::is_trivially_copy_constructible<T>::value << '\n';
    std::cout << "is nothrow copy-constructible: ";
    std::cout << std::is_nothrow_copy_constructible<T>::value << "\n\n";

    std::cout << "is move-constructible: ";
    std::cout << std::is_move_constructible<T>::value << '\n';
    std::cout << "is trivially move-constructible: ";
    std::cout << std::is_trivially_move_constructible<T>::value << '\n';
    std::cout << "is nothrow move-constructible: ";
    std::cout << std::is_nothrow_move_constructible<T>::value << "\n\n";

    std::cout << "is copy-assignable: ";
    std::cout << std::is_copy_constructible<T>::value << '\n';
    std::cout << "is trivially copy-assignable: ";
    std::cout << std::is_trivially_copy_constructible<T>::value << '\n';
    std::cout << "is nothrow copy-assignable: ";
    std::cout << std::is_nothrow_copy_constructible<T>::value << "\n\n";

    std::cout << "is move-assignable: ";
    std::cout << std::is_move_assignable<T>::value << '\n';
    std::cout << "is trivially move-assignable: ";
    std::cout << std::is_trivially_move_assignable<T>::value << '\n';
    std::cout << "is nothrow move-assignable: ";
    std::cout << std::is_nothrow_move_assignable<T>::value << "\n\n";
}

void test_copy_constructible()
{
    display_copy_constructible<Ex1>("Ex1");
    display_copy_constructible<Ex2>("Ex2");
    display_copy_constructible<Ex3>("Ex3");
    display_copy_constructible<NoMove>("NoMove");
    display_copy_constructible<NoMove_noexcept>("NoMove_noexcept");
    display_copy_constructible<OnlyMove>("OnlyMove");
    display_copy_constructible<OnlyMove_noexcept>("OnlyMove_noexcept");
    display_copy_constructible<HasMove>("HasMove");
    display_copy_constructible<HasMove_noexcept>("HasMove_noexcept");
    display_copy_constructible<HasMove_both_noexcept>("HasMove_both_noexcept");

    std::cout << '\n';
}

void test_move_constructible()
{
    display_move_constructible<Ex1>("Ex1");
    display_move_constructible<Ex2>("Ex2");
    display_move_constructible<Ex3>("Ex3");
    display_move_constructible<NoMove>("NoMove");
    display_move_constructible<NoMove_noexcept>("NoMove_noexcept");
    display_move_constructible<OnlyMove>("OnlyMove");
    display_move_constructible<OnlyMove_noexcept>("OnlyMove_noexcept");
    display_move_constructible<HasMove>("HasMove");
    display_move_constructible<HasMove_noexcept>("HasMove_noexcept");
    display_move_constructible<HasMove_both_noexcept>("HasMove_both_noexcept");

    std::cout << '\n';
}

/*******************************************************

  output:

    Ex1 is move-constructible? true
    Ex1 is trivially move-constructible? false
    Ex1 is nothrow move-constructible? true

    Ex2 is move-constructible? true
    Ex2 is trivially move-constructible? true
    Ex2 is nothrow move-constructible? true

    Ex3 is move-constructible? true
    Ex3 is trivially move-constructible? false
    Ex3 is nothrow move-constructible? true

    NoMove is move-constructible? true
    NoMove is trivially move-constructible? false
    NoMove is nothrow move-constructible? false

 *******************************************************/

void test_copy_assignable()
{
    display_copy_assignable<Ex1>("Ex1");
    display_copy_assignable<Ex2>("Ex2");
    display_copy_assignable<Ex3>("Ex3");
    display_copy_assignable<NoMove>("NoMove");
    display_copy_assignable<NoMove_noexcept>("NoMove_noexcept");
    display_copy_assignable<OnlyMove>("OnlyMove");
    display_copy_assignable<OnlyMove_noexcept>("OnlyMove_noexcept");
    display_copy_assignable<HasMove>("HasMove");
    display_copy_assignable<HasMove_noexcept>("HasMove_noexcept");
    display_copy_assignable<HasMove_both_noexcept>("HasMove_both_noexcept");

    std::cout << '\n';
}

void test_move_assignable()
{
    display_move_assignable<Ex1>("Ex1");
    display_move_assignable<Ex2>("Ex2");
    display_move_assignable<Ex3>("Ex3");
    display_move_assignable<NoMove>("NoMove");
    display_move_assignable<NoMove_noexcept>("NoMove_noexcept");
    display_move_assignable<OnlyMove>("OnlyMove");
    display_move_assignable<OnlyMove_noexcept>("OnlyMove_noexcept");
    display_move_assignable<HasMove>("HasMove");
    display_move_assignable<HasMove_noexcept>("HasMove_noexcept");
    display_move_assignable<HasMove_both_noexcept>("HasMove_both_noexcept");

    std::cout << '\n';
}

/*******************************************************

  output:

    Ex1 is move-assignable? true
    Ex1 is trivially move-assignable? false
    Ex1 is nothrow move-assignable? true

    Ex2 is move-assignable? false
    Ex2 is trivially move-assignable? false
    Ex2 is nothrow move-assignable? false

    Ex3 is move-assignable? false
    Ex3 is trivially move-assignable? false
    Ex3 is nothrow move-assignable? false

    NoMove is move-assignable? true
    NoMove is trivially move-assignable? true
    NoMove is nothrow move-assignable? true

 *******************************************************/

void test_copy_and_move_all()
{
    display_copy_and_move_all<Ex1>("Ex1");
    display_copy_and_move_all<Ex2>("Ex2");
    display_copy_and_move_all<Ex3>("Ex3");
    display_copy_and_move_all<NoMove>("NoMove");
    display_copy_and_move_all<NoMove_noexcept>("NoMove_noexcept");
    display_copy_and_move_all<OnlyMove>("OnlyMove");
    display_copy_and_move_all<OnlyMove_noexcept>("OnlyMove_noexcept");
    display_copy_and_move_all<HasMove>("HasMove");
    display_copy_and_move_all<HasMove_noexcept>("HasMove_noexcept");
    display_copy_and_move_all<HasMove_both_noexcept>("HasMove_both_noexcept");
}
 
int main()
{
    std::cout << "------------------------------------------------";
    std::cout << '\n';

    test_copy_constructible();
    test_move_constructible();
    test_copy_assignable();
    test_move_assignable();

    std::cout << "------------------------------------------------";
    std::cout << '\n';

    test_copy_and_move_all();
    return 0;
}
