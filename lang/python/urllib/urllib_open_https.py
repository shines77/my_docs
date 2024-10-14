#
# From: https://zhuanlan.zhihu.com/p/684518663
#
import urllib.request
import ssl

f = open("example.com.html", "wb")

context = ssl.create_default_context()
# 如遇到SSL验证错误, 可以忽略证书验证
context.check_hostname = False
context.verify_mode = ssl.CERT_NONE

url = 'https://example.com/'
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
with urllib.request.urlopen(req, context=context) as response:
    html_content = response.read()
    html_content_utf8 = html_content.decode('utf-8')
    print(html_content_utf8)
    f.write(html_content)

f.close()
