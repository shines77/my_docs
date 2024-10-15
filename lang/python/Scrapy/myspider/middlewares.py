
import random
#from scrapy.contrib.downloadermiddleware.useragent import UserAgentMiddleware
from scrapy.downloadermiddlewares.useragent import UserAgentMiddleware

from settings import USER_AGENT_LIST

class RotateUserAgentMiddleware(UserAgentMiddleware):
    '''
    用户代理中间件（处于下载中间件位置）
    '''

    def process_request(self, request, spider):
        user_agent = random.choice(USER_AGENT_LIST)
        if user_agent:
            request.headers.setdefault('User-Agent', user_agent)
            print(f"User-Agent: {user_agent}")
