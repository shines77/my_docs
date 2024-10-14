#
# From: https://zhuanlan.zhihu.com/p/684518663
#
import urllib.request

f = open("example.com.html", "wb")

url = 'http://example.com'
with urllib.request.urlopen(url) as response:
    html_content = response.read()
    html_content_utf8 = html_content.decode('utf-8')
    print(html_content_utf8)
    f.write(html_content)

f.close()
