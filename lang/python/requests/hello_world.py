
import requests

response = requests.get('https://api.github.com/user', auth=('user', 'pass'))

print(response.status_code)
print(response.headers['content-type'])
print(response.encoding)
print(response.text)
print(response.json())
