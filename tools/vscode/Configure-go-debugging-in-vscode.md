# 在 VSCode 中配置 go 调试

## 1. 安装和配置

用 `VSCode` 打开你的 `go` 项目所在的文件夹，或者 `go` 源代码文件，打开菜单：`运行(R)` -> `启动调试`，或者直接按 `F5` 键。`VSCode` 会提示你安装 `dlv`，还会安装别的一些东西，都一一安装即可。

此时，默认是没有调试配置的，你可以点开 `VSCode` 左边栏上方的绿色小三角的下拉列表，最后有一项是 “`添加设置...`” ，在弹出的菜单里找到 “`{} GO: Launch Package`” ，添加配置，会新增一段 `json` 代码，把它修改为如下所示：

```js
{
    // 使用 IntelliSense 了解相关属性。 
    // 悬停以查看现有属性的描述。
    // 欲了解更多信息，请访问: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Launch Package",
            "type": "go",
            "request": "launch",
            "mode": "debug",
            "program": "C:\\Project\\Golang\\hashmap-benchmark\\",
            "env": {},
            "args": []
        }
    ]
}
```

`"program"` 一项填入你的 `go` 项目的 `package` 所在的文件夹，则调试整个 `package`，如果 `"program"` 填入的是单个 `go` 文件，则只调试单个文件。

然后，使用 `F9` 设置断点，按下 `F5` 即可开始进入调试。

## 2. 其他

如果在 “`调试控制台`” 中显示如下红色的错误信息：

```bash
Version of Go is too old for this version of Delve (minimum supported version 1.12, suppress this error with --check-go-version=false)
Process exiting with code: 1
```

说明你的 `golang` 版本低于 `1.12`，更新到新版本即可。

## 3. 参考文章

* [1]. [用vscode开发调试golang超简单教程](https://blog.csdn.net/v6543210/article/details/84504460) From `blog.csdn.net`

* [2]. [在vscode中怎样debug调试go程序](https://www.cnblogs.com/ljhoracle/p/11047083.html) From `www.cnblogs.com`
