# Conda 使用教程

## 1. MiniConda

推荐使用 Miniconda, 安装包更小, 默认库比较少, 可以自由选择.

## 2. conda 相关命令

创建新的conda环境:

```
conda create --name python3
或
conda create --name python3 python=3.12.4
```

列出环境列表:

```
conda env list
```

```
# conda environments:
#
base                  *  C:\ProgramData\miniconda3
python3                  C:\Users\guoxi\.conda\envs\python3
scrapy                   C:\Users\guoxi\.conda\envs\scrapy
```

激活环境:

```
conda activate scrapy
```

删除 env:

```
conda env remove --name python3
```

env 改名:

```
conda rename --name python3 new_env_name
```

## 3. 参考文章

- [conda如何创建虚拟环境](https://www.php.cn/faq/630707.html)
