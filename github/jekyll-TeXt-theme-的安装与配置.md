# jekyll-TeXt-theme 的安装与配置

## 1. 快速开始

### 1.1 安装开发环境

如果你想在本地调试或运行主题，你需要安装 `Ruby` 以及 `Jekyll`，详情请戳 [这里](https://jekyllrb.com/docs/installation/) 。

`Windows` 下安装与配置，请参考这里:

[https://jekyllrb.com/docs/installation/windows/](https://jekyllrb.com/docs/installation/windows/)

### 1.2 普通方式安装

克隆 `git` 仓库：

```bash
git clone https://github.com/shines77/shines77.github.io.git
```

上述步骤完成后，安装 `Ruby` 依赖包：

```bash
# 切换到 git 仓库的根目录
cd ./shines77.github.io

bundle config set --local path 'vendor/bundle'
bundle install
```

旧版 `bundle` 的命令格式是：

```bash
bundle install --path vendor/bundle
```

### 1.3 本地预览

`Jekyll` 集成了一个开发用的服务器，可以让你使用浏览器在本地进行预览。

使用下面的命令启动 `Jekyll` 开发服务器：

```bash
bundle exec jekyll serve -P 8000
```

稍等十秒左右，你就可以通过 `http://127.0.0.1:8000/` 预览你的网站了。

`错误信息`

如果显示如下错误信息：

```shell
`require': cannot load such file -- webrick (LoadError)
```

原因是，由于 `Ruby 3.0.0` 以上不再自带 `WebRick`，需要手动添加到环境里面。

`解决方法`

将 `webrick` 添加到依赖当中:

```bash
bundle add webrick
```

这个要在项目中执行。

如果 `webrick` 不存在，可以使用 `gem` 安装：

```bash
gem install webrick
```
