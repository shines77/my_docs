# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy

class Game4399Item(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()

    # 这里定义了三个字段，分别表示游戏的名称、类别和日期
    name = scrapy.Field()
    category = scrapy.Field()
    date = scrapy.Field()
    # new_field = scrapy.Field()

# 可以定义其他字段来表示不同的信息
class OtherItem(scrapy.Item):
    pass
