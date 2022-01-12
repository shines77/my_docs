# Sublime Text 3 中如果安装 Rust 语法高亮支持

## 1. 安装 Package Control

### 1.1 使用命令安装

打开 Sublime Text 3，打开 “命令面板”，快捷键如下：

* Windows / Linux：Ctrl + Shift + P
* Mac：Cmd + Shift + P

或者，使用菜单进入 “命令面板”：

1. 打开 “工具” 菜单项；
2. 选择 “命令面板...”。

然后输入 "Install Package Control"，按回车。

这会下载最新版本的 “Package Control”，并使用公钥验证。如果出现错误，则使用手动的方法代替。

英文原文：This will download the latest version of Package Control and verify it using public key cryptography. If an error occurs, use the manual method instead.

### 1.2 手动安装

If the command palette/menu method is not possible due to a proxy on your network or using an old version of Sublime Text, the following steps will also install Package Control:

1. Click the Preferences > Browse Packages… menu
2. Browse up a folder and then into the Installed Packages/ folder
3. Download Package Control.sublime-package and copy it into the Installed Packages/ directory
4. Restart Sublime Text

或者使用下面的代码手动安装：

Sublime Text 3 编辑器中使用 “Ctrl + ~” 快捷键启动控制台（不一定能调出来），粘贴以下代码，并回车进行安装。

```java
import urllib.request,os,hashlib; h = '2915d1851351e5ee549c20394736b442' + '8bc59f460fa1548d1514676163dafc88'; pf = 'Package Control.sublime-package'; ipp = sublime.installed_packages_path(); urllib.request.install_opener( urllib.request.build_opener( urllib.request.ProxyHandler()) ); by = urllib.request.urlopen( 'http://packagecontrol.io/' + pf.replace(' ', '%20')).read(); dh = hashlib.sha256(by).hexdigest(); print('Error validating download (got %s instead of %s), please try manual install' % (dh, h)) if dh != h else open(os.path.join( ipp, pf), 'wb' ).write(by)
```

## 2. 安装 Rust 语法高亮插件

按 “Ctrl + Shift + P” 调出 “命令面板”，然后输入 “Install Package”，如果有多个选择，则选择“Package Control: Install Package”。

然后经过短暂的联网搜索（枚举），会列出所有可选的插件包，如果再次弹出输入框，则输入 “Rust”，自动完成列表里应该能看到如下的插件包：

```shell
Rust Enhanced
RustFormat
RustAutoComplete
```

推荐安装 “Rust Enhanced”，这是官方支持的 `Rust` 语法高亮包。Enjoy it !

## 3. 参考文章

1. [RustPrime: Sublime Text 3](https://rustcc.gitbooks.io/rustprimer/content/editors/sublime.html)

2. [packagecontrol.io: Package Control](https://packagecontrol.io/installation#Manual)
