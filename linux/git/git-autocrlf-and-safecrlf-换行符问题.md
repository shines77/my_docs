# 如何解决 Git 中的 AutoCRLF 与 SafeCRLF 换行符问题

`Git` 中的源码可能包含 `Windows` 下的 `CrLf` 字符，或者 `Linux` 下的是 `Lf` 字符，设置为 “git config --global core.autocrlf true” 后，在提交转换为 `Lf`，检出时根据当前系统决定转换为 `CrLf`，或者不转换。

## 1. AutoCRLF

关于配置的作用域，`systemwide` > `global` > `local`。`local` 如果没有配置，`global` 的设置为 “autocrlf = false”，`systemwide` 的设置为 “autocrlf = true”，则实际生效的设置是 `global`，即 “autocrlf = false”；如果 `global` 中也没有配置，则实际生效的设置是 `systemwide`，即 “autocrlf = true”。

```bash
# 提交时转换为 LF，检出时根据当前系统决定转换为 CRLF，或者不转换
git config --global core.autocrlf true

# 提交时转换为 LF，检出时不转换
git config --global core.autocrlf input

# 提交检出时均不转换, 保持原样
git config --global core.autocrlf false
```

默认值为 “core.autocrlf input”，在 `TortoiseGit` 中的该选项名称为：“自动换行符转化”，勾选为"true"，不勾选为 "false"，半勾选为 "input"，可以自己手动编辑全局 `.git/config` 文件。

## 2. SafeCRLF

```bash
# 拒绝提交包含混合换行符的文件
git config --global core.safecrlf true

# 允许提交包含混合换行符的文件
git config --global core.safecrlf false

# 提交包含混合换行符的文件时给出警告
git config --global core.safecrlf warn
```

默认值为 “core.safecrlf false”，在 `TortoiseGit` 中的该选项名称为：“检查换行”，这三个选项都有。

## 3. 推荐设置

在 `Windows` 系统上推荐使用：

```bash
# 提交时转换为 LF，检出时根据当前系统决定转换为 CRLF，或者不转换
git config --global core.autocrlf true

# 拒绝提交包含混合换行符的文件
git config --global core.safecrlf true
```

## 4. 参考文章

+ `[1]`. [git换行符之AutoCRLF配置的意义](https://www.cnblogs.com/yepei/p/5650290.html)

+ `[2]`. [如何解决Git中的AutoCRLF与SafeCRLF换行符问题](https://www.yisu.com/zixun/553593.html)
