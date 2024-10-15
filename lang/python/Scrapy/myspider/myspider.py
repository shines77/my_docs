#
# From: https://scrapy.org/
#

import scrapy

class BlogSpider(scrapy.Spider):
    name = 'blogspider'
    start_urls = ['https://www.zyte.com/blog/']

    def parse(self, response):
        for title_h2 in response.css('h2[class="mt-24 heading-medium"]'):
            title = title_h2.css('a::text')
            yield {'title': title.css('::text').get()}

        for next_page in response.css('a.next'):
            yield response.follow(next_page, self.parse)

#
# shell command:
# scrapy runspider myspider.py
#