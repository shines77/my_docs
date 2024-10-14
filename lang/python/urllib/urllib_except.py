#
# From: https://zhuanlan.zhihu.com/p/684518663
#

import urllib.request
from urllib.error import URLError, HTTPError

url = 'http://invalid-url.com'

try:
    with urllib.request.urlopen(url) as response:
        html_content = response.read()
        print(html_content)
except URLError as ex:
    print('We failed to reach a server. URL = [%s]' % url)
    print('Reason:', ex.reason)
except HTTPError as ex:
    if ex.code == 404:
        print('Error code: 404, Not found')
    else:
        print('The server couldn\'t fulfill the request.')
        print('Error code: ', ex.code)
        print('Error message: ', ex.read())
except ContentTooShortError as ex:
    # 当从 URL 读取的内容比预期的要短时，会抛出此异常。
    # 这通常在使用 urlopen 与文件类对象（如 urlopen(url, data)）一起使用时发生。
    print('Content too short error.')
except RequestError as ex:
    # 当请求中发生错误时，会抛出此异常。它通常不直接由用户处理，而是作为其他更具体错误的基类
    print('An error occurred in the request.')
