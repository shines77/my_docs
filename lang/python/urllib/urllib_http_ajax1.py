#
# From: https://blog.csdn.net/weixin_57023347/article/details/132780256
#

import urllib.parse
import urllib.request

# 创建 Request 对象
def create_request(page):
    # base_url = 'https://movie.douban.com/j/chart/top_listtype=5&interval_id=100%3A90&action=&'
    base_url = "https://movie.douban.com/j/search_subjects?type=movie&tag=%E7%83%AD%E9%97%A8&"

    url_args = {
        'page_start': (page - 1) * 20,
        'page_limit': 20
    }
    url_arg = urllib.parse.urlencode(url_args)
    url = base_url + url_arg
    print('url = ', url)

    # 避免反爬验证
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.3029.110 Safari/537.36'
    }

    request = urllib.request.Request(url=url, headers=headers)
    return request

# 获取 HTML 正文
def get_html_content(request):
    response = urllib.request.urlopen(request)
    html_content = response.read().decode('utf-8')
    print(html_content)
    return html_content

# 保存 json 内容
def save_json_data(page, content):
    # 需要自己在当前目录下新建一个 douban 的目录
    fp = open('./douban/douban_' + str(page) + '.json', 'w', encoding='utf-8')
    fp.write(content)

def main():
    start_page = int(input('请输入开始页: '))
    end_page = int(input('请输入结束页: '))

    for page in range(start_page, end_page + 1):
        request = create_request(page)
        html_content = get_html_content(request)
        save_json_data(page, html_content)

if __name__ == '__main__':
    main()
