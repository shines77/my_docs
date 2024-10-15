# -*- coding: utf-8 -*-
#
# From: https://scrapy.org/
#
import sys
# sys.setdefaultencoding('utf-8')

import scrapy

class BlogSpider(scrapy.Spider):
    name = "blogspider"
    allowed_domains = [
        "zyte.com"
    ]
    start_urls = ['https://www.zyte.com/blog/']

    '''
    def start_requests(self):
        yield scrapy.Request('https://www.zyte.com/blog/', self.parse, encoding='utf-8')

    def parse(self, response):
        print('len(response.text) = ', len(response.text))
        pass

        for h2 in response.xpath("//h2").getall():
            yield { "title": h2 }

        for href in response.xpath("//a/@href").getall():
            yield scrapy.Request(response.urljoin(href), self.parse)

        # title_h2 = response.css('h2[class="mt-24 heading-medium"]')
        # title_h2 = response.xpath('//h2')
        #return

        for title_h2 in response.css('h2[class="mt-24 heading-medium"]'):
            title_a = title_h2.css('a')
            yield {'title': title_a.css('::text').get()}

        for next_page in response.css('a.next'):
            yield response.follow(next_page, self.parse)
        '''

    def parse(self, response):
        self.logger.info("Hi, this is an item page! %s", response.url)
        print('len(response.text) = ', len(response.text))
        item = scrapy.Item()
        print(response.headers.get('content-type', '').lower())
        # enc = if 'charset' in response.headers.get('content-type', '').lower() else None
        url = response.url
        encoding = response.encoding
        print('response.encoding = ', response.encoding)
        # html_content = response.content.decode(encoding, 'replace') if encoding else response.text
        # item["h2"] = response.selector.xpath(b'//h2[@class="mt-24 heading-medium"]/text()').get()
        # print('item["h2"] = ', item["h2"])
        # item["link_text"] = response.meta["link_text"]
        # print('item["link_text"] = ', item["link_text"])
        item_h2_a = response.xpath('//h2[@class="mt-24 heading-medium"]//a/text()').getall()
        # item_h2 = response.xpath('//h2').getall()
        # print('item_h2 = ', item_h2)
        for title in item_h2_a:
            print('title = ', title)
        return response.follow(
            url, self.parse_additional_page, cb_kwargs=dict(item=item)
        )

    def parse_additional_page(self, response, item):
        #item["additional_data"] = response.xpath(
        #    '//p[@id="additional_data"]/text()'
        #).get()
        return item
