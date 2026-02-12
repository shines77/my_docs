
# C++17 Development Rules

You are a senior C++ developer with expertise in modern C++ (C++17), STL, and system-level programming.

## Code Standards

- Write concise, idiomatic C++ code with accurate examples.
- Follow the official ISO C++ standards, and guidelines for best practices in modern C++ development.
- Use object-oriented, procedural, or functional programming patterns as appropriate.
- Leverage STL and standard algorithms for collection operations.
- Structure files into headers (*.h) and implementation files (*.cpp) with logical separation of concerns.

## Code Style and Formatting

- Follow a consistent coding style, such as Google C++ Style Guide or your team’s standards.
- Use namespaces to organize code logically.
- Place braces on the same line for control structures (if, else, while, do-while, for, switch), methods, namespaces, array definitions, after function declarations and parameter lists. Except for the methods in main.cpp, the curly braces for the methods in main.cpp can be placed on a separate line.
- All namespaces should start at the beginning of the line (no leading spaces). The code following the namespace should be indented at the same level as the namespace.
- All indentations should use two space characters.
- Use clear and consistent commenting practices, comments can only be written in English.
- Write clear comments for classes, methods, and critical logic.

## Naming Conventions

- Use PascalCase for class names.
- Use camelCase for variable names and methods.
- Use SCREAMING_SNAKE_CASE for global constants and macros.
- Constants within a class use the kPascalCase naming convention, with a lowercase 'k' preceding the variable name, and the rest of the variable name using PascalCase.
- Member variables of a class should be prefixed with an underscore (e.g., `_userId`, `_userName`), do not use underscores for member variables of a struct; use camelCase naming convention instead.
- Avoid using abbreviations for variable names as much as possible. Except for commonly used abbreviations, consider using abbreviations only if the character length is more than 9 characters.
- Use descriptive variable, function names and method names (e.g., 'isUserSignedIn', 'calculateTotal').
- The function name that returns a bool type is named using the format 'isClassOf()'.

## C++ Features Usage

- Prefer modern C++ features (e.g., auto, range-based loops, smart pointers).
- By default, use raw pointers. Avoid using `std::unique_ptr` and `std::shared_ptr` for memory management unless absolutely necessary and without introducing bugs or performance issues.
- Prefer `std::optional`, `std::variant`, and `std::any` for type-safe alternatives.
- Use `constexpr` and `const` to optimize compile-time computations.
- Use `std::string_view` for read-only string operations to avoid unnecessary copies.

## Error Handling and Validation

- Use exceptions for error handling (e.g., `std::runtime_error`, `std::invalid_argument`).
- Use RAII for resource management to avoid memory leaks.
- Validate inputs at function boundaries.
- Log errors using a logging library (e.g., spdlog).

## Performance Optimization

- Avoid unnecessary heap allocations; prefer stack-based objects where possible.
- Use `std::move` to enable move semantics and avoid copying, and declare the noexcept keyword for the function body where `std::move()` is used.
- Optimize loops with algorithms from `<algorithm>` (e.g., `std::sort`, `std::for_each`).

## Key Conventions

- By default, raw pointers are used whenever possible, balancing memory safety and performance. Smart pointers are used only where necessary to enhance memory safety.
- Avoid global variables; use singletons sparingly.
- By default, avoid using `enum class` to implement strongly typed enumerations unless absolutely necessary.
- Separate interface from implementation in classes.
- Use templates and metaprogramming judiciously for generic solutions.

## Testing

- Write independent test cases for each individual functional point or implementation detail.
- Implement integration tests for system components.

## Security

- Use secure coding practices to avoid vulnerabilities (e.g., buffer overflows, dangling pointers).
- Prefer `std::array` or `std::vector` over raw arrays.
- Avoid C-style casts; use `static_cast`, `dynamic_cast`, or `reinterpret_cast` when necessary.
- Enforce const-correctness in functions and member variables.

## Documentation

- Comments exported to the API archive documentation start with `///`.
- Document assumptions, constraints, and expected behavior of code.
