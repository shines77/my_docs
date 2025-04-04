#
# From: https://blog.csdn.net/weixin_57023347/article/details/132780256
#

import urllib.request
from urllib.request import ProxyHandler, build_opener
import random

proxies = [
    {'http': '127.0.0.1:1080'},
    {'http': '127.0.0.1:1081'},
    {'http': '127.0.0.1:8080'},
    {'http': '127.0.0.1:8081'}
]

# 随机选一个代理
proxy = random.choice(proxies)
print(proxy)

# 得到 Proxy handler 对象
proxy_handler = ProxyHandler({'http': 'http://proxy.example.com:8080'})

# 构建 opener 对象
opener = build_opener(proxy_handler)

url = 'http://example.com'

# 获得响应
with opener.open(url) as response:
    html_content = response.read()
    html_content_utf8 = html_content.decode('utf-8')
    print(html_content_utf8)
