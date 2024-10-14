#
# From: https://zhuanlan.zhihu.com/p/684518663
#

import urllib.request

url = 'http://example.com/redirect'
with urllib.request.urlopen(url) as response:
    redirected_url = response.geturl()
    html_content = response.read()
    print(redirected_url)
    print(html_content)
