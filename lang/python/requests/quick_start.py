#
# From: https://requests.readthedocs.io/en/latest/user/quickstart/#make-a-request
#

import requests
import json

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
}

## Http Get 1
payload = {
    'key1': 'value1',
    'key2': 'value2'
}

r = requests.get('https://httpbin.org/get', headers=headers, params=payload)
print(r.url)
print(r.status_code)
# https://httpbin.org/get?key2=value2&key1=value1
print(r.status_code == requests.codes.ok)

## Http Get 2
payload = {
    'key1': 'value1',
    'key2': ['value2', 'value3']
}

r = requests.get('https://httpbin.org/get', headers=headers, params=payload)
print(r.url)
print(r.status_code)
# https://httpbin.org/get?key1=value1&key2=value2&key2=value3

## Http Post 1
post_data = {
    'key1': 'value1',
    'key2': 'value2'
}

r = requests.post('https://httpbin.org/post', headers=headers, data=post_data)
print(r.url)
print(r.status_code)
print(r.text)
'''
{
  ...
  "form": {
    "key2": "value2",
    "key1": "value1"
  },
  ...
}
'''

## Http Post 2
payload_dict = {
    'key1': 'value1',
    'key2': ['value2', 'value3']
}

r = requests.post('https://httpbin.org/post', headers=headers, data=payload_dict)
print(r.url)
print(r.status_code)
print(r.text)
'''
{
  ...
  "form": {
    "key1": "value1",
    "key2": [
      "value2",
      "value3"
    ]
  },
  ...
}
'''

## Http Post Json
if False:
    # Please note that the below code will NOT add the Content-Type header
    # (so in particular it will NOT set it to application/json).
    url = 'https://api.github.com/some/endpoint'
    payload_json = {
        'some': 'data'
    }

    r = requests.post(url, data=json.dumps(payload_json))
    # It's same to the above code (added in version 2.4.2)
    r = requests.post(url, json=payload)

## Http others
print('https://httpbin.org/put')
r = requests.put('https://httpbin.org/put', headers=headers, data={'key': 'value'})
print(r.status_code)

print('https://httpbin.org/delete')
r = requests.delete('https://httpbin.org/delete', headers=headers)
print(r.status_code)

print('https://httpbin.org/get')
r = requests.head('https://httpbin.org/get', headers=headers)
print(r.status_code)

print('https://httpbin.org/get')
r = requests.options('https://httpbin.org/get', headers=headers)
print(r.status_code)

## Response Content
print('https://api.github.com/events')
r = requests.get('https://api.github.com/events', headers=headers)
# This is text, not the original binary content
print(r.text)
print(r.status_code)
print(r.encoding)
# '[{"repository":{"open_issues":0,"url":"https://github.com/...
# 'utf-8'

# Binary Response Content
from PIL import Image
from io import BytesIO

#r = requests.get('https://e-assets.gitee.com/gitee-community-web/_next/static/media/logo-white.a5b0e29c.svg')
r = requests.get('https://foruda.gitee.com/avatar/1676896153886685230/15580_shines77_1608656637.png')
# The gzip and deflate transfer-encodings are automatically decoded for you.
img = Image.open(BytesIO(r.content))

with open('avatar.png', 'wb+') as fd:
    for chunk in r.iter_content(chunk_size=128):
        fd.write(chunk)

## JSON Response Content
r = requests.get('https://api.github.com/events')
print(r.json())
print(r.status_code)
print(r.encoding)

## Raw Response Content
r = requests.get('https://api.github.com/events', stream=True)

print(r.raw)
# <urllib3.response.HTTPResponse object at 0x101194810>

print(r.raw.read(10))
# b'\x1f\x8b\x08\x00\x00\x00\x00\x00\x00\x03'

with open('github-event.json', 'wb+') as fd:
    for chunk in r.iter_content(chunk_size=128):
        fd.write(chunk)

## POST a Multipart-Encoded File
url = 'https://httpbin.org/post'
files = {
    'file': open('report.xls', 'rb')
}

r = requests.post(url, files=files)
print(r.text)
'''
{
  ...
  "files": {
    "file": "<censored...binary...data>"
  },
  ...
}
'''

url = 'https://httpbin.org/post'
files = {
    'file': ('report.xls', open('report.xls', 'rb'), 'application/vnd.ms-excel', {'Expires': '0'})
}

r = requests.post(url, files=files)
print(r.text)
'''
{
  ...
  "files": {
    "file": "<censored...binary...data>"
  },
  ...
}
'''

#
# If you want, you can send strings to be received as files:
#
url = 'https://httpbin.org/post'
files = {
    'file': ('report.csv', 'some,data,to,send\nanother,row,to,send\n')
}

r = requests.post(url, files=files)
print(r.text)
'''
{
  ...
  "files": {
    "file": "some,data,to,send\\nanother,row,to,send\\n"
  },
  ...
}
'''

## Response Headers
if False:
    print(r.headers)
    '''
    {
        'content-encoding': 'gzip',
        'transfer-encoding': 'chunked',
        'connection': 'close',
        'server': 'nginx/1.0.4',
        'x-runtime': '148ms',
        'etag': '"e1ca502697e5c9317743dc078f67693f"',
        'content-type': 'application/json'
    }
    '''

    print(r.headers['Content-Type'])
    print(r.headers.get('content-type'))
    '''
    'application/json'
    '''

## Cookies
if False:
    # 1. If a response contains some Cookies, you can quickly access them.
    url = 'http://example.com/some/cookie/setting/url'
    r = requests.get(url)

    print(r.cookies['example_cookie_name'])
    '''
    'example_cookie_value'
    '''

    # 2. To send your own cookies to the server, you can use the cookies parameter.
    url = 'https://httpbin.org/cookies'
    cookies = dict(cookies_are='working')

    r = requests.get(url, cookies=cookies)
    print(r.text)
    '''
    '{"cookies": {"cookies_are": "working"}}'
    '''

    # 3. Cookies are returned in a RequestsCookieJar,
    # which acts like a dict but also offers a more complete interface,
    # suitable for use over multiple domains or paths.
    # Cookie jars can also be passed in to requests.
    jar = requests.cookies.RequestsCookieJar()

    jar.set('tasty_cookie', 'yum', domain='httpbin.org', path='/cookies')
    jar.set('gross_cookie', 'blech', domain='httpbin.org', path='/elsewhere')

    url = 'https://httpbin.org/cookies'
    r = requests.get(url, cookies=jar)
    print(r.text)
    '''
    '{"cookies": {"tasty_cookie": "yum"}}'
    '''

## Redirection and History
if False:
    r = requests.get('http://github.com/')
    print(r.url)
    # 'https://github.com/'
    print(r.status_code)
    # 200
    print(r.history)
    # [<Response [301]>]

    r = requests.get('http://github.com/', allow_redirects=False)
    print(r.url)
    # 'https://github.com/'
    print(r.status_code)
    # 301
    print(r.history)
    # []

## Timeout
if False:
    r = requests.get('https://github.com/', timeout=0.001)
    '''
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    requests.exceptions.Timeout: HTTPConnectionPool(host='github.com', port=80): Request timed out. (timeout=0.001)
    '''

## Errors and Exceptions
if False:
    # In the event of a network problem (e.g. DNS failure, refused connection, etc),
    # Requests will raise a ConnectionError exception.

    # Will raise an HTTPError if the HTTP request returned an unsuccessful status code.
    Response.raise_for_status()

    # If a request times out, a Timeout exception is raised.

    # If a request exceeds the configured number of maximum redirections, a TooManyRedirects exception is raised.

    # All exceptions that Requests explicitly raises inherit from requests.exceptions.RequestException.
