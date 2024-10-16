# scrapy 基本命令

有一个细节, 不能用 `conda install` 的方式安装 `scrapy`, `lxml` 有可能出现 `Unknown encoding` 编码错误.

在一个干净的 `conda` 环境下, 直接使用 `pip install scrapy` 即可, 这样不会报错.

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

其中 `{your_spider_name}` 和 `{your_project_name}` 不能同名, 但创建后可以自己改名.

例如:

```
scrapy genspider blog_spider zyte.com
```

这样会在项目文件夹中新建一个 ./{your_project_name}/{your_spider_name}.py 的爬虫主体文件.

例如: ./blogspider/blog_spider.py

## 3. 运行 scrapy 爬虫程序

然后:

```
scrapy crawl {your_spider_name}
```

注意, 这里启动的是你的 `{spider_name}`, 而不是 `{project_name}`.

例如:

```
scrapy crawl blog_spider
```

## 4. 运行单个文件的 scrapy 代码

```
scrapy runspider myscrapy
```

运行当前目录下的 myscrapy.py 爬虫程序.
