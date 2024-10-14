#
# From: https://zhuanlan.zhihu.com/p/684518663
#

from urllib.parse import parse_qs, parse_qsl

query_string = 'key1=value1&key2=value2'

# 返回一个字典，键是唯一的，如果有多个相同的键，则它们的值是一个列表
query_dict = parse_qs(query_string)
print(query_dict)  # {'key1': ['value1'], 'key2': ['value2']}

# 返回一个列表，每个元素都是一个元组，包含键和值
query_list = parse_qsl(query_string)
print(query_list)  # [('key1', 'value1'), ('key2', 'value2')]

# ----------------------------------------------

from urllib.parse import urlencode

## 构建查询字符串

params = {'key1': 'value1', 'key2': 'value2'}
query_string = urlencode(params)

print(query_string)  # key1=value1&key2=value2
