
count = 0                   # 计数器
user_name = "admin"         # 登录用户名
user_password = "12345"     # 登录密码

f = open("lock_users.txt", "r")
file_list = f.readlines()
f.close()

lock_users = []

name = input("登录用户名：")
for i in file_list:1
    line = i.strip("\n")
    lock_users.append(line)

if name in lock_users:
    print("你的账户已锁定，请联系管理员。")
else:
    if name == user_name:
        # 如果密码连续输错了三次，锁定账号
        while count < 3:
            password = input("登录密码：")
            if name == user_name and password == user_password:
                print("欢迎 %s !" % name)
                break
            else:
                print("账号和密码不匹配!")
                count += 1
        else:
            print("对不起，您的账号连续输错三次密码已被锁定，请联系管理员。")
            f = open("lock_users.txt", "w+")
            li = ['%s' % user_name]
            f.writelines(li)
            f.close()
    else:
        print("用户名不存在，请输入正确的用户名。")