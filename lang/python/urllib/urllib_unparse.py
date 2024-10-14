#
# From: https://zhuanlan.zhihu.com/p/684518663
#

from urllib.parse import urlunparse

parts = ('http', 'www.example.com', '/path', None, 'name=value', 'fragment')
url = urlunparse(parts)

print("url = %s" % url)  # http://www.example.com/path?name=value#fragment
