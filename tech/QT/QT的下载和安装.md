# QT的下载和安装

## 1. 官方下载

Qt 官网有一个专门的资源下载网站，所有的开发环境和相关工具都可以从这里下载，具体地址是：[http://download.qt.io/](http://download.qt.io/) 。

目录结构说明：

- **archive**: 各种 Qt 开发工具安装包，新旧都有（可以下载 Qt 开发环境和源代码）。
- **community_releases**: 社区定制的 Qt 库，Tizen 版 Qt 以及 Qt 附加源码包。
- **development_releases**: 开发版，有新的和旧的不稳定版本，在 Qt 开发过程中的非正式版本。
- **learning**: 有学习 Qt 的文档教程和示范视频。
- **ministro**: 迷你版，目前是针对 Android 的版本。
- **official_releases**: 正式发布版，是与开发版相对的稳定版 Qt 库和开发工具（可以下载Qt开发环境和源代码）。
- **online**: Qt 在线安装源。
- **snapshots**: 预览版，最新的开发测试中的 Qt 库和开发工具。

Qt 的历史版本在 `archive` 目录，但不是所有版本都能下载，部分版本已无法下载，例如：5.1 到 5.12 之间的版本被删除了。

较新的版本在 `official_releases` 目录中，但只提供最近的几个版本。

子目录的说明：

- **vsaddin**: 这是 Qt 针对 Visual Studio 集成的插件.
- **qtcreator**: 这是 Qt 官方的集成开发工具，但是 qtcreator 本身是个空壳，它没有编译套件和 Qt 开发库。除了老版本的 Qt 4 需要手动下载 qtcreator、编译套件、Qt 开发库进行搭配之外，一般用不到。Qt 5 有专门的大安装包，里面包含开发需要的东西，并且能自动配置好。
- **qt**: 这是 Qt 开发环境的下载目录，我们刚说的 Qt 5 的大安装包就在这里面。
- **online_installers**: 在线安装器，国内用户不建议使用，在线安装是龟速，还经常断线。

## 2. 下载镜像

这里给大家推荐几个国内著名的 Qt 镜像网站，主要是各个高校的：

```bash
清华大学：https://mirrors.tuna.tsinghua.edu.cn/qt/
北京理工大学：http://mirror.bit.edu.cn/qtproject/
中国科学技术大学：http://mirrors.ustc.edu.cn/qtproject/
```

国内镜像网站的结构和官方是类似的，但是跟官方比，是不完整的，像 `qt-opensource-windows-x86-5.15.0.exe` 这种就无法下载了，只能下载 `/single` 目录下的 `qt-everywhere-src-5.15.0.zip` 。

## 3. 参考文章

- [全网最全的Qt下载途径（多种下载通道+所有版本）](https://www.cnblogs.com/kn-zheng/p/17689855.html)
