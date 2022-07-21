# Git: submodule 子模块简明手册

## 1. 添加子模块

使用如下命令：

```bash
git submodule add -b master https://github.com/shines77/jstd.git ./3rd_party/jstd
或
git submodule add --force -b master -- "https://gitee.com/shines77/emhash.git" "./3rd_party/emhash"
```

添加以后，会出现 `.gitmodule` 文件，内容如下：

```ini
[submodule "3rd_party/jstd"]
	path = 3rd_party/jstd
	url = https://gitee.com/shines77/jstd.git
	branch = master
```

`.git/config` 中也会有如下信息：

```ini
[submodule "3rd_party/jstd"]
	url = https://gitee.com/shines77/jstd.git
```

同时，`.git/modules` 目录下也会出现 `jstd` 目录。

## 2. 更新子模块

先初始化 `submodule` ，然后更新到主项目设置的 `子模块` 版本：

```bash
git submodule init
git submodule update --init --recursive
```

如果想更新到 `子模块` 最新的版本，请使用：

```bash
git submodule update --remote --recursive
```

## 3. 删除子模块

如果添加错了 `子模块`，或者想更换子模块的远端地址，可以先把 `子模块` 删掉。

删除 `子模块` 比较麻烦，需要手动删除相关的文件，否则在添加子模块时有可能出现错误。

1. 删除子模块的文件夹

```bash
git rm --cached ./3rd_party/jstd
```

2. 删除 `.gitmodules` 文件中相关子模块的信息，例如：

```ini
[submodule "3rd_party/jstd"]
	path = 3rd_party/jstd
	url = https://gitee.com/shines77/jstd.git
	branch = master
```

3. 删除 `.git/config` 文件中相关子模块的信息，例如：

```ini
[submodule "3rd_party/jstd"]
	url = https://gitee.com/shines77/jstd.git
```

4. 删除 `.git/modules` 文件夹中的相关子模块的文件夹，例如：

```bash
rm -rf ./.git/modules/jstd
```

## 4. 查看子模块

查看子模块当前的版本状态，例如：

```bash
$ git submodule

 2e7bb1e4b34c14df4c917edfd4cbad879f9935ad 3rd_party/jstd (remotes/origin/HEAD)
 2e36d96669b0f56eaf865245c7bcf8060963a088 3rd_party/abseil-cpp (heads/master)
 12ebab8e31cc2d50dd870c72a52a2813f4d1f252 3rd_party/flat_hash_map (heads/master)
```

## 5. 参考文章

1. `[Git: submodule 子模块简明教程]`

    [https://zhuanlan.zhihu.com/p/404615843](https://zhuanlan.zhihu.com/p/404615843)

2. `[git submodule update --init 和 --remote 的区别]`

    [https://blog.csdn.net/fanyun_01/article/details/115338145](https://blog.csdn.net/fanyun_01/article/details/115338145)
