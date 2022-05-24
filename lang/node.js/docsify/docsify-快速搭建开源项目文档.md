# Docsify 快速搭建开源项目文档

## 1. Docsify

搭建开源项目文档的工具，一般有 Docsify、VuePress、Hexo、Jelly 和 GitBook 等。

本文只介绍 `Docsify` 如果搭建搭建开源项目文档。

`Docsify` 是一个动态生成文档网站的工具。不同于 VuePress、GitBook、Hexo 的地方是它不会生成将 .md 转成 .html 文件，所有转换工作都是在运行时进行。

这将非常实用，如果只是需要快速的搭建一个小型的文档网站，或者不想因为生成的一堆 .html 文件 "污染" commit 记录，只需要创建一个 `index.html` 就可以开始写文档而且直接部署在 `GitHub Pages` 。

`Docsify` 官网：[https://docsify.js.org/](https://docsify.js.org/)

## 2. 安装 Node.js

安装 Node.js：

Node.js 官网：[https://nodejs.org/zh-cn/](https://nodejs.org/zh-cn/)

由于 node.js 官网访问比较慢，可以使用国内的镜像：[http://nodejs.cn/](http://nodejs.cn/)

下载页面：[http://nodejs.cn/download/](http://nodejs.cn/download/)

## 3. 解决 npm 下载慢的问题

因为 `npm` 是从国外服务器下载的，可能速度很慢。可以改用国内的 `npm` 镜像。

例如：

```shell
npm install -g cnpm --registry=https://registry.npm.taobao.org
```

安装完成以后，执行 `cnmp -v` 命令会显示如下错误信息：

```shell
cnpm : 无法加载文件 C:\Users\XXXXXX\AppData\Roaming\npm\cnpm.ps1，因为在此系统上禁止运行脚本。
```

这是因为 `Windows PowerShell` 的 `*.ps1` 脚本运行策略的问题，解决办法是：

1. 以管理员身份运行 `Windows PowerShell`；

2. 运行命令：

    ```shell
    get-ExecutionPolicy
    ```

    若显示 `Restricted` 则表示运行状态是禁止的，若显示 `RemoteSigned` 则表示 OK 了。

3. 执行命令：

    ```shell
    set-ExecutionPolicy
    ```

    会提示你输入参数，输入 `RemoteSigned` 回车，即可。如果之后还提示进行选择，输入 `Y`  回车。

    如果没有以 “管理员身份运行” 运行 `Windows PowerShell`，此步会报错。

4. 再次使用 `get-ExecutionPolicy` 命令检查一下设置是否成功。

## 4. 安装 docsify

`docsify` 官方快速开始文档：[https://docsify.js.org/#/quickstart](https://docsify.js.org/#/quickstart)

因为之前安装了淘宝的源 `cnpm`，所以下面的 `npm` 命令可以替换为 `cnpm` 。

快速开始

创建 docsify 文档根目录：

```shell
mkdir .\docsify-demo   # Windows
或
mkdir ./docsify-demo   # Linux
```

全局安装 docsify-cli 工具：

```shell
npm i docsify-cli -g
```

初始化项目：

```shell
docsify init ./docs
```

初始化成功后，可以看到 `./docs` 目录下创建的几个文件：

* index.html：入口文件
* README.md：会做为主页内容渲染
* .nojekyll：用于阻止 GitHub Pages 忽略掉下划线开头的文件

直接编辑 `docs/README.md` 就能更新文档内容，当然也可以添加更多页面。

预览网站：

```shell
docsify serve docs

Serving C:\Project\nodejs\docsify-demo\docs now.
Listening at http://localhost:3000
```

用浏览器打开 `http://localhost:3000`，预览效果。

官网文档范本：

由于并没有具体的文档文件，网站首页空空如也。

我们可以把 `docsify` github 的仓库拉下来（下载 zip 文件即可），地址如下：

```shell
https://github.com/docsifyjs/docsify
```

并把其中 .\docs 目录的内容拷贝我们的 `docsify` 项目下面的 .\docs 目录，再初始化项目，即可得到和 docsify 官网一模一样的文档。

## 5. 部署到服务器

`docsify` 部署文档：[https://docsify.js.org/#/deploy](https://docsify.js.org/#/deploy)

可以部署到如下网站：

* GitHub Pages
* GitLab Pages
* Firebase Hosting
* VPS
* Netlify
* ZEIT Now
* AWS Amplify

## 6. 自定义配置

`docsify` 自定义配置文档：[https://docsify.js.org/#/configuration](https://docsify.js.org/#/configuration)

所有配置都在 `./docs/index.html` 文件中的 `window.$docsify` 里。

如下：

```html
<!-- index.html -->

<!DOCTYPE html>
<html>
<head>
  <meta http-equiv="X-UA-Compatible" content="IE=edge,chrome=1">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <meta charset="UTF-8">
  <link rel="stylesheet" href="//cdn.jsdelivr.net/npm/docsify@4/themes/vue.css" />
</head>
<body>
  <div id="app"></div>
  <script>
    window.$docsify = {
      //...
    }
  </script>
  <script src="//cdn.jsdelivr.net/npm/docsify@4"></script>
</body>
</html>
```

### 6.1 name, repo

* `name`：文档标题，显示在侧栏顶部。
* `nameLink`：点击文档标题后，跳转的链接地址（可以是相对路径）。
* `repo`：GitHub 仓库地址，页面右上角会渲染一个 GitHub 角标。

```js
name: 'docsify',
nameLink: '/',
repo: 'https://github.com/shines77/docsify-demo/',
```

`nameLink` 可以设置为如下格式，更完整：

```js
nameLink: {
    '/': '#/',
    '/es/': '#/es/',
    '/de-de/': '#/de-de/',
    '/ru-ru/': '#/ru-ru/',
    '/zh-cn/': '#/zh-cn/',
},
```

### 6.2 coverpage

* coverpage：设置是否启用封面页，默认不启用。若开启，加载的是项目根目录下的 `_coverpage.md` 文件。

虽然 `docsify` 的封面不错，但有时候我们不需要它，设置为 `false` 可以关闭封面。

```js
coverpage: false,
```

### 6.3 loadSidebar

加载自定义侧边栏，具体可以参考：[https://docsify.js.org/#/more-pages](https://docsify.js.org/#/more-pages) 。

```js
loadSidebar: true,
```

这个选项在官网的 `docsify` 默认是打开的，因为 `docsify` 就是带侧边栏的。

### 6.4 _sidebar.md

增加 `_sidebar.md` 文件，文件格式如下：

```yaml
- [CentOS](centos.md)
- [Docker](docker.md)
- [Mac](mac.md)
- [NPM](npm.md)
- [推荐](recommend.md)
```

### 6.5 subMaxLevel, maxLevel

* subMaxLevel：自定义侧边栏后，默认不会再生成目录，设置生成目录的最大层级（建议配置为2-4）。
* maxLevel：最大支持渲染的标题层级

```js
subMaxLevel: 2,
maxLevel: 4,
```

配合 `loadSidebar`，实现自定义侧边栏的二级目录 。

### 6.6 loadNavbar

* loadNavbar：导航栏支持，默认加载的是项目根目录下的 `_navbar.md` 文件。
* mergeNavbar：小屏设备下合并导航栏到侧边栏。

```js
loadNavbar: true,
mergeNavbar: true,
```

### 6.7 auto2top

切换页面后是否自动跳转到页面顶部。

```js
auto2top: true,
```

### 6.8 routerMode

设置路由模式。

如果设置为 `history` 之后，浏览器链接里不会出现 `#`，可能会对 `SEO` 更友好，看你的个人习惯。

默认设置为 ‘hash’ ，或者删除 `routerMode` 这一项。

```js
routerMode: 'hash',
```

修改了这个值之后，需要使用 `docsify serve docs` 重新启动网站，否则可能会报错。

## 7. docsify 插件

`docsify` 插件文档：[https://docsify.js.org/#/plugins](https://docsify.js.org/#/plugins)

`docsify` 有丰富的插件，可以按需添加。

### 7.1 Full text search

全局搜索：

```js
search: {
  noData: {
    '/es/': '¡No hay resultados!',
    '/de-de/': 'Keine Ergebnisse!',
    '/ru-ru/': 'Никаких результатов!',
    '/zh-cn/': '没有结果!',
    '/': 'No results!',
  },
  placeholder: {
    '/es/': 'Buscar',
    '/de-de/': 'Suche',
    '/ru-ru/': 'Поиск',
    '/zh-cn/': '搜索',
    '/': 'Search',
  },
  paths: 'auto',
  // 搜索标题的最大层级, 1 - 6
  depth: 6,
  pathNamespaces: ['/es', '/de-de', '/ru-ru', '/zh-cn'],
},
```

开启全局搜索需要引入两个 `js` 文件：

```html
<script src="//cdn.jsdelivr.net/npm/docsify@4/lib/docsify.min.js"></script>
<script src="//cdn.jsdelivr.net/npm/docsify@4/lib/plugins/search.min.js"></script>
```

### 7.2 Copy to Clipboard

复制到剪贴板，在所有的代码块上添加一个简单的 `Copy to Clipboard` 按钮来允许用户从你的文档中复制代码。

需要引入 `js` 文件：

```html
<script src="//cdn.jsdelivr.net/npm/docsify-copy-code@4"></script>
```

### 7.3 分页导航 - Pagination

分页导航，在文档的最下方会展示上一个文档和下一个文档。

```js
pagination: {
  previousText: '上一章节',
  nextText: '下一章节',
  crossChapter: true,
  crossChapterText: false,
},
```

需要引入两个 `js` 文件：

```html
<script src="//cdn.jsdelivr.net/npm/docsify@4/lib/docsify.min.js"></script>
<script src="//cdn.jsdelivr.net/npm/docsify-pagination@2/dist/docsify-pagination.min.js"></script>
```

### 7.4 图片缩放 - Zoom image

Medium's 风格的图片缩放插件. 基于 [medium-zoom](https://github.com/francoischalifour/medium-zoom) 。

需要引入 `js` 文件：

```html
<script src="//cdn.jsdelivr.net/npm/docsify@4/lib/plugins/zoom-image.min.js"></script>
```

### 7.5 emoji

默认是提供 `emoji` 解析的，能将类似 `:100:` 解析成 :100: 。但是它不是精准的，因为没有处理非 `emoji` 的字符串。如果你需要正确解析 `emoji` 字符串，你可以引入这个插件。

需要引入 `js` 文件：

```html
<script src="//cdn.jsdelivr.net/npm/docsify@4/lib/plugins/emoji.min.js"></script>
```

## 8. 参考文章

1. `Docsify快速搭建个人博客`

    [https://www.imooc.com/article/287154](https://www.imooc.com/article/287154)

2. `Wiki系列（二）：docsify部署及配置`

    [https://juemuren4449.com/archives/docsify-deploy-and-configuration](https://juemuren4449.com/archives/docsify-deploy-and-configuration)

3. `入坑 docsify，一款神奇的文档生成利器！`

    [https://baijiahao.baidu.com/s?id=1683928475208184783](https://baijiahao.baidu.com/s?id=1683928475208184783)

4. `Docsify使用指南（打造最快捷、最轻量级的个人&团队文档）`

    [https://www.cnblogs.com/Can-daydayup/p/15413267.html](https://www.cnblogs.com/Can-daydayup/p/15413267.html)

5. `解决 windows “因为在此系统上禁止运行脚本报错”`

    [https://blog.csdn.net/qq_47183158/article/details/120088725](https://blog.csdn.net/qq_47183158/article/details/120088725)
