#
# From : https://blog.csdn.net/sixpp/article/details/137457648
#

import urllib.request

#--------------------------------------------------------
# 发送 HTTP 请求

url_http = 'http://www.example.com'
http_headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'
}

req_http = urllib.request.Request(url_http, headers=http_headers)
response_http = urllib.request.urlopen(req_http)
html_http = response_http.read().decode('utf-8')
print(html_http)
print("--------------------------------------------------------")

#--------------------------------------------------------
# 发送 HTTPS 请求

url_https = 'https://www.example.com'
https_headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'
}

req_https = urllib.request.Request(url_https, headers=https_headers)
response_https = urllib.request.urlopen(req_https)
html_https = response_https.read().decode('utf-8')
print(html_https)
print("--------------------------------------------------------")
