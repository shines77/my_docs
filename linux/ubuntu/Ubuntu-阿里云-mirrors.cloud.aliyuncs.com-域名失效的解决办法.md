# Ubuntu 下阿里云 mirrors.cloud.aliyuncs.com 域名失效的解决办法

## 1. 现象

当执行 `apt update` 的时候，显示：

```shell
Err:1 http://mirrors.cloud.aliyuncs.com/ubuntu focal InRelease
  Could not resolve 'mirrors.cloud.aliyuncs.com'

W: Failed to fetch http://mirrors.cloud.aliyuncs.com/ubuntu/dists/focal/InRelease  Could not resolve 'mirrors.cloud.aliyuncs.com'
```

注：`mirrors.cloud.aliyuncs.com` 这个域名需要在阿里云 `云服务器`，`轻量应用服务器` 等 `vpc` 下才能解析。

## 2. 原因

系统内部配置的 `dns` 服务器（例如：`127.0.0.53`）无法解析域名 `mirrors.cloud.aliyuncs.com` 导致的。
或者其他依赖的主机导致 `mirrors.cloud.aliyuncs.com` 无法解析。

## 3. 解决办法

### 3.1 备份

```shell
cp  /etc/resolv.conf  /etc/resolv.conf.bak
```

### 3.2 修复

由于 `Ubuntu` 从某个版本开始，`/etc/resolv.conf` 是由几个模版文件拼接生成的，分别叫 head, tail, base, original 等。直接修改 `/etc/resolv.conf` 即刻生效，但是重启以后会被重新覆盖，所以我们在修改这个文件：

```shell
vim /etc/resolvconf/resolv.conf.d/head
````

在注释的下一行，加入如下内容：

```shell
options timeout:1 attempts:1 rotate
nameserver 100.100.2.136
nameserver 100.100.2.138
```

同时在把这三行添加到 `/etc/resolv.conf` 前面，因为 `head` 文件要下次重启才会生效，对 `/etc/resolv.conf` 的修改是即刻生效的，改完就可以。

注：`100.100.2.136`，`100.100.2.138` 是阿里云内部的 `DNS` 服务器。

## 4. 验证测试

```shell
dig mirrors.cloud.aliyuncs.com
```

看到下面字样就说明解析成功了：

```shell
;; QUESTION SECTION:
;mirrors.cloud.aliyuncs.com.	IN	A

;; ANSWER SECTION:
mirrors.cloud.aliyuncs.com. 600	IN	A	100.100.2.148
```

用 `apt update` 验证一下更新是否正常了。

## 5. 参考文章

1. [【云安全中心】Linux 更新软件-域名解析失败](https://developer.aliyun.com/article/767805)
