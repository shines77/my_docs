#
# From: https://zhuanlan.zhihu.com/p/684518663
#

from urllib.parse import urlparse

url = 'http://www.example.com/path?name=value#fragment'
parsed = urlparse(url)

print("parsed.scheme   = %s" % parsed.scheme)    # http
print("parsed.netloc   = %s" % parsed.netloc)    # www.example.com
print("parsed.path     = %s" % parsed.path)      # /path
print("parsed.params   = %s" % parsed.params)    # None
print("parsed.query    = %s" % parsed.query)     # name=value
print("parsed.fragment = %s" % parsed.fragment)  # fragment
