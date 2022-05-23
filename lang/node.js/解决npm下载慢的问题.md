# 解决 npm 下载慢的问题

## 1. 解决 npm 下载慢的问题

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

## 2. 参考文章

1. `[入坑 docsify，一款神奇的文档生成利器！]`

    [https://baijiahao.baidu.com/s?id=1683928475208](https://baijiahao.baidu.com/s?id=1683928475208)

2. `[解决 windows “因为在此系统上禁止运行脚本报错”]`

    [https://blog.csdn.net/qq_47183158/article/details/120088725](https://blog.csdn.net/qq_47183158/article/details/120088725)

