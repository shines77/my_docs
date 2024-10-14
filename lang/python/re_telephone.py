
import re

texts = {
    '(123)-456-7890',
    '123-456-7890',
    '(123-456-7890',
    '123)-456-7890'
}

def get_telephone(text):
    # (123)-456-7890 或者 123-456-7890
    # pattern = r'\(\d{3}\)-\d{3}-\d{4}|^\d{3}-\d{3}-\d{4}$'
    pattern = r'\(\d{3}\)-\d{3}-\d{4}|\d{3}-\d{3}-\d{4}'
    telephone = re.search(pattern, text)
    if telephone:
        telephone = telephone.group()
    result = '{} 匹配到的电话号码为: {}'.format(text, telephone)
    print(result)

# text = input('请输入字符串: ')
for text in texts:
    get_telephone(text)
