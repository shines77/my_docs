#
# From: https://zhuanlan.zhihu.com/p/684518663
#

import urllib.request
from urllib.request import ProxyHandler, build_opener

proxy_handler = ProxyHandler({'http': 'http://proxy.example.com:8080'})
opener = build_opener(proxy_handler)

url = 'http://example.com'
with opener.open(url) as response:
    html_content = response.read()
    html_content_utf8 = html_content.decode('utf-8')
    print(html_content_utf8)
