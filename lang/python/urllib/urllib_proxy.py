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
    print(html_content)
