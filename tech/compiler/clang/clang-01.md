# clang 4.0.1 分析 -- 预处理命令


## 预处理命令的处理

由于预处理命令关键字不多，只有十几个关键字，所以采用比较简单快速的 `Hash` 公式，配合 `switch-case` 来实现。

`Hash 算法`采取：`关键长度` + `关键字第一个字符` + `关键字第三个字符` 组成。

（注：唯一一个两个字符的关键字 `if`，第三个字符取 `'\0'`，也是非常精妙。）

具体实现请参考下面代码：

源文件：`\llvm\tools\clang\lib\Basic\IdentifierTable.cpp`<br/>
类名：`class IdentifierInfo`

```cpp
tok::PPKeywordKind IdentifierInfo::getPPKeywordID() const {
  // We use a perfect hash function here involving the length of the keyword,
  // the first and third character.  For preprocessor ID's there are no
  // collisions (if there were, the switch below would complain about duplicate
  // case values).  Note that this depends on 'if' being null terminated.

#define HASH(LEN, FIRST, THIRD) \
  (LEN << 5) + (((FIRST-'a') + (THIRD-'a')) & 31)
#define CASE(LEN, FIRST, THIRD, NAME) \
  case HASH(LEN, FIRST, THIRD): \
    return memcmp(Name, #NAME, LEN) ? tok::pp_not_keyword : tok::pp_ ## NAME

  unsigned Len = getLength();
  if (Len < 2) return tok::pp_not_keyword;
  const char *Name = getNameStart();
  switch (HASH(Len, Name[0], Name[2])) {
  default: return tok::pp_not_keyword;
  CASE( 2, 'i', '\0', if);
  CASE( 4, 'e', 'i', elif);
  CASE( 4, 'e', 's', else);
  CASE( 4, 'l', 'n', line);
  CASE( 4, 's', 'c', sccs);
  CASE( 5, 'e', 'd', endif);
  CASE( 5, 'e', 'r', error);
  CASE( 5, 'i', 'e', ident);
  CASE( 5, 'i', 'd', ifdef);
  CASE( 5, 'u', 'd', undef);

  CASE( 6, 'a', 's', assert);
  CASE( 6, 'd', 'f', define);
  CASE( 6, 'i', 'n', ifndef);
  CASE( 6, 'i', 'p', import);
  CASE( 6, 'p', 'a', pragma);

  CASE( 7, 'd', 'f', defined);
  CASE( 7, 'i', 'c', include);
  CASE( 7, 'w', 'r', warning);

  CASE( 8, 'u', 'a', unassert);
  CASE(12, 'i', 'c', include_next);

  CASE(14, '_', 'p', __public_macro);

  CASE(15, '_', 'p', __private_macro);

  CASE(16, '_', 'i', __include_macros);
#undef CASE
#undef HASH
  }
}
```

