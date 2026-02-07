# node.js 切换 npm 源的几种方式

## 1. 在 Windows 下

### 1.1 修改 .npmrc 文件

修改全局 .npmrc 文件（对所有项目有效）：

```bash
registry=https://registry.npmmirror.com/
```

### 1.2 npm 命令

通过命令行更改当前项目的源：

```bash
npm config set registry https://registry.npmmirror.com/
```

查看当前源的命令:

```bash
npm config get registry
```

## 2. NRM 包切换

安装依赖:

```
npm install -g nrm
```

查看可用的源:

```
nrm ls
```

```
$ nrm ls
  npm ---------- https://registry.npmjs.org/
  yarn --------- https://registry.yarnpkg.com/
  tencent ------ https://mirrors.cloud.tencent.com/npm/
  cnpm --------- https://r.cnpmjs.org/
* taobao ------- https://registry.npmmirror.com/
  npmMirror ---- https://skimdb.npmjs.com/registry/
  huawei ------- https://repo.huaweicloud.com/repository/npm/
```

切换源:

```
nrm use taobao
```

测试速度（所有源的响应时间）:

根据源的响应时间长短选择最快的源使用即可。

```
nrm test
```

## 3. 解决 npm 下载慢的问题

(下面这种方法已失效)

因为 `npm` 是从国外服务器下载的，可能速度很慢。可以改用国内的 `npm` 镜像。

例如：

```shell
npm install -g cnpm --registry=https://registry.npmmirror.com
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

1. `[Nodejs切换源的两种方式]`

    [https://blog.csdn.net/2301_79943136/article/details/137815212](https://blog.csdn.net/2301_79943136/article/details/137815212)

2. `[入坑 docsify，一款神奇的文档生成利器！]`

    [https://baijiahao.baidu.com/s?id=1683928475208184783](https://baijiahao.baidu.com/s?id=1683928475208184783)

3. `[解决 windows “因为在此系统上禁止运行脚本报错”]`

    [https://blog.csdn.net/qq_47183158/article/details/120088725](https://blog.csdn.net/qq_47183158/article/details/120088725)
