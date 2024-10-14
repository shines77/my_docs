#
# From: https://blog.csdn.net/weixin_57023347/article/details/132780256
#

import urllib.parse
import urllib.request

url = 'http://example.com/'

get_args = {
    'wd': '周杰伦',
    'sex': '男',
    'location': '台湾'
}

get_arg = urllib.parse.urlencode(get_args)
print(get_arg)

url = url + get_arg
print(url)

with urllib.request.urlopen(url) as response:
    html_content = response.read()
    html_content_utf8 = html_content.decode('utf-8')
    print(html_content_utf8)
