#
# From: https://zhuanlan.zhihu.com/p/684518663
#

import urllib.request
import urllib.parse

http_headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
}

# 登录url(post请求)
# post_url = 'http://example.com/post'
post_url = 'https://passport.baidu.com/v6/api/?login'

args = {
    "username": "15798017910",
    "password": "ExCK7PIdLq3CyPdOJUQ+YK/mrTIFEvbIWK1tQd+XDBMR2lgpH1Ri9CyhOfX67/DW4Y/9JlFtFYU"
}

# 构建 Post 参数
post_data = urllib.parse.urlencode(args).encode('utf-8')

request = urllib.request.Request(url=post_url, headers=http_headers, data=post_data, method='POST')

with urllib.request.urlopen(request) as response:
    html_content = response.read().decode('utf-8')
    print(html_content)
