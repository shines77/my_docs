#
# From: https://zhuanlan.zhihu.com/p/684518663
#

import urllib.request
import urllib.parse

url = 'http://example.com/post'
data = urllib.parse.urlencode(
    {'key1': 'value1', 'key2': 'value2'}
).encode()

request = urllib.request.Request(url, data, method='POST')
with urllib.request.urlopen(request) as response:
    html_content = response.read()
    print(html_content)
