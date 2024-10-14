#
# From: https://zhuanlan.zhihu.com/p/684518663
# From: https://blog.csdn.net/sixpp/article/details/137457648
#

import urllib.robotparser

# 创建一个 RobotFileParser 实例
rp = urllib.robotparser.RobotFileParser()

# 设置要解析的 robots.txt 文件的 URL（通常是一个本地文件路径或网络 URL）
# 这里我们使用一个示例的字符串来代替实际的文件内容
robots_txt_content = """
User-agent: *
Disallow: /private/
Disallow: /temp/

User-agent: Googlebot
Allow: /temp/
"""

# 使用 set_url 方法来设置要解析的内容，通常这是一个 URL，但也可以是字符串内容
rp.set_url('https://www.baidu.com/robots.txt')
rp.read()
# lines = ""
# robots_txt_content = rp.parse(lines)
#rp.parse(lines)
#print(lines)

print("rp.mtime() = ", rp.mtime())
print("rp.modified() = ", rp.modified())

rrate = rp.request_rate("Googlebot")
if rrate != None:
    print("rrate = rp.request_rate('Googlebot')")
    print("rrate.requests = ", rrate.requests)
    print("rrate.seconds  = ", rrate.seconds)

print("rp.crawl_delay('Googlebot') = ", rp.crawl_delay("Googlebot"))

# 检查特定用户代理是否可以访问特定路径
user_agent = "Googlebot"
path = "http://www.baidu.com/temp/somefile.html"
if rp.can_fetch(user_agent, path):
    print(f"{user_agent} is allowed to fetch {path}")
else:
    print(f"{user_agent} is not allowed to fetch {path}")

# 对于不在 robots.txt 中明确允许的路径，默认是允许的
path = "http://www.baidu.com/public/index.html"
if rp.can_fetch(user_agent, path):
    print(f"{user_agent} is allowed to fetch {path} (by default)")
else:
    print(f"{user_agent} is not allowed to fetch {path} (by default)")

path = "http://www.baidu.com/baidu"
if rp.can_fetch(user_agent, path):
    print(f"{user_agent} is allowed to fetch {path} (by default)")
else:
    print(f"{user_agent} is not allowed to fetch {path} (by default)")
