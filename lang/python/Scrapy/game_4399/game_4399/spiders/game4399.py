#
# From: https://blog.csdn.net/2301_77659011/article/details/135630678
#
import scrapy
from game_4399.items import Game4399Item

class Game4399Spider(scrapy.Spider):
    name = "game4399"
    allowed_domains = ["4399.com"]
    start_urls = ["https://4399.com/flash/"]

    def parse(self, response):
        # print(response)  # <200 http://www.4399.com/flash/>
        # print(response.text)  # 打印页面源代码
        # response.xpath()  # 通过xpath解析数据
        # response.css()  # 通过css解析数据

        # 获取4399小游戏的游戏名称
        # txt = response.xpath('//ul[@class="n-game cf"]/li/a/b/text()')
        # txt 列表中的每一项是一个Selector：
        # <Selector query='//ul[@class="n-game cf"]/li/a/b/text()' data='逃离克莱蒙特城堡'>]
        # 要通过extract()方法拿到data中的内容
        # print(txt)

        # txt = response.xpath('//ul[@class="n-game cf"]/li/a/b/text()').extract()
        # print(txt)  # 此时列表中的元素才是游戏的名字

        # 也可以先拿到每个li, 然后再提取名字
        li_list = response.xpath('//ul[@class="n-game cf"]/li')
        for li in li_list:
            # name = li.xpath('./a/b/text()').extract()
            # # name 是一个列表
            # print(name)  # ['王城霸业']

            # 一般我们都会这么写：li.xpath('./a/b/text()').extract()[0]
            # 但是这样如果列表为空就会报错, 所以换另一种写法
            # extract_first方法取列表中的第一个, 如果列表为空, 返回None
            name = li.xpath('./a/b/text()').extract_first()         # 游戏名称: 王城霸业
            category = li.xpath('./em/a/text()').extract_first()    # 游戏类别
            date = li.xpath('./em/text()').extract_first()          # 日期
            # print(name)
            # print(category, date)

            '''
            # 通过 yield 向管道传输数据
            dic = {
                'name': name,
                'category': category,
                'date': date
            }

            # 可以认为这里是把数据返回给了管道 pipeline,
            # 但是实际上是先给引擎, 然后引擎再给管道, 只是这个过程不用我们关心, scrapy会自动完成.
            # 这里的数据会在管道程序中接收到.
            yield dic
            '''

            # 我们现在不返回字典, 而是返回真正推荐我们返回的格式之一：item
            # 先导入GameItem类：from game.items import GameItem
            # 然后创建它的实例, 使用起来和字典类似
            # 区别就是GameItem类里没有定义的字段, 就不能使用, 比如不能item['某个没有定义的字段']
            item = Game4399Item()
            # item['xxx'] 里的 xxx 要在类 Game4399Item 里定义, 否则就会报错
            item['name'] = name
            item['category'] = category
            item['date'] = date
            yield item
