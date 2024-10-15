# scrapy 基本命令

## 1. 创建 scrapy 项目

```
scrapy startproject {your_project_name}
```

例如:

```
scrapy startproject blogspider
```

这样会新建一个叫 `blogspider` 的项目目录.

## 2. 创建一个爬虫程序

先切换到刚才创建的项目目录:

cd {your_project_name}

然后:

```
scrapy genspider {your_spider_name} {your_domain_name}
```

例如:

```
scrapy genspider blogspider zyte.com
```

这样会在项目文件夹中新建一个 ./{your_project_name}/{your_spider_name}.py 的爬虫主体文件.

例如: ./blogspider/blogspider.py

## 3. 运行 scrapy 爬虫程序

然后:

```
scrapy crawl {your_project_name}
```

例如:

```
scrapy crawl blogspider
```

## 4. 运行单个文件的 scrapy 代码

```
scrapy runspider myscrapy
```

运行当前目录下的 myscrapy.py 爬虫程序.
