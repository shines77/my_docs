##
## From: https://drissionpage.cn/get_start/examples/control_browser
##

import time, os
from time import sleep

from DrissionPage import Chromium
from DrissionPage import ChromiumOptions

co = ChromiumOptions()
co.set_argument('--start-maximized')

# 阻止“自动保存密码”的提示气泡
co.set_pref('credentials_enable_service', False)

# 阻止“要恢复页面吗？Chrome未正确关闭”的提示气泡
co.set_argument('--hide-crash-restore-bubble')

browser = Chromium(co)

##
## Html 定位语法
## https://drissionpage.cn/browser_control/get_elements/syntax
##

# 启动或接管浏览器，并创建标签页对象
tab = browser.latest_tab

# 跳转到登录页面
tab.get('https://gitee.com/login')

# 定位到账号文本框，获取文本框元素
ele = tab.ele('#user_login')

# 对文本框输入账号
ele.input('wokss@163.com')

# 定位到密码文本框并输入密码
ele = tab.ele('#user_password')

# 对文本框输入密码 (这里密码需要自己填)
ele.input('xxxxxxxxxx')

tab.ele('#user_remember_me').click()
time.sleep(1.0)

# 点击登录按钮
tab.ele('@value=登 录').click()

time.sleep(1000)
browser.quit()


## 如何在无界面 Linux 使用

##
## DrissionPage在deb系Linux的使用
## https://zhuanlan.zhihu.com/p/674687748
##

if False:
    from DrissionPage import ChromiumPage, ChromiumOptions

    co = ChromiumOptions()
    co.set_argument('--start-maximized')

    # 阻止“自动保存密码”的提示气泡
    co.set_pref('credentials_enable_service', False)

    # 阻止“要恢复页面吗？Chrome未正确关闭”的提示气泡
    co.set_argument('--hide-crash-restore-bubble')

    co = co.set_argument('--no-sandbox')    # 关闭沙箱模式, 解决`$DISPLAY`报错
    co = co.set_headless(True)              # 开启无头模式, 解决`浏览器无法连接`报错

    page = ChromiumPage(co)
    page.get('http://DrissionPage.cn')
    print(page.title)
