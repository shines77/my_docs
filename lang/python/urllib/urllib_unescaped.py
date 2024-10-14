#
# From: https://zhuanlan.zhihu.com/p/684518663
#

from urllib.parse import quote, unquote

# 转义
escaped_string = quote('Hello, World!')
print(escaped_string)  # Hello%2C%20World%21

# 解转义
unescaped_string = unquote(escaped_string)
print(unescaped_string)  # Hello, World!
