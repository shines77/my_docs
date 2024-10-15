
# Windows终端改默认编码

## 1. 通过命令行修改

```bash
chcp 65001
```

## 2. 通过 PowerShell 更改

```bash
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
```

## 3. 参考文章

- [修改windows终端cmd控制台默认编码为utf-8](https://www.cnblogs.com/luckyang/p/18269484)
