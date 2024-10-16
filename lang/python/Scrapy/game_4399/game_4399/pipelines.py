# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface
from itemadapter import ItemAdapter
import pymysql

# 导入 MySQL 配置
from game_4399.settings import MYSQL

class Game4399Pipeline:
    def __init__(self):
        self.f = None

    def open_spider(self, spider):
        print('[Game4399Pipeline]: 爬虫开启了...')
        self.f = open('./game_data.csv', mode='a', encoding='utf-8')

    def process_item(self, item, spider):
        print('[Game4399Pipeline]: 爬虫进行中...')

        """
        接收爬虫通过引擎传递过来的数据
        :param item: 具体的数据内容
        :param spider: 对应传递数据的爬虫程序
        :return:
        """
        print(item)     # {'name': '武器升级', 'category': '休闲类', 'date': '2024-01-06'}
        # print(spider)   # <Game4399Spider 'game_4399' at 0x22867dafc70>

        try:
            self.f.write(f'{item["name"]}, {item["category"]}, {item["date"]}\n')
        except:
            self.logger.error(f'Writting file error: {item["name"]}, {item["category"]}, {item["date"]}\n')
            if self.f:
                self.f.close
                self.f = None
        finally:
            pass

        return item

    def close_spider(self, spider):
        print('[Game4399Pipeline]: 爬虫关闭了...')
        if self.f:
            self.f.close()


# 默认情况下管道是不开启的, 需要在 settings.py 文件中进行设置
class Game4399MysqlPipeline:
    def __init__(self):
        self.conn = None

    def open_spider(self, spider):
        print('[Game4399MysqlPipeline]: 爬虫开启了...')

        # 更好的办法是写在 settings 文件中
        # 然后从 settings 文件中导入：from game.settings import MYSQL
        # 创建数据库连接
        self.conn = pymysql.connect(
            host = MYSQL['host'],           # 主机
            port = MYSQL['port'],           # 端口
            user = MYSQL['user'],           # 用户名
            password = MYSQL['password'],   # 密码
            database = MYSQL['database']    # 数据库名称
        )

    def process_item(self, item, spider):
        """
        接收爬虫通过引擎传递过来的数据
        :param item: 具体的数据内容
        :param spider: 对应传递数据的爬虫程序
        :return:
        """
        print('[Game4399MysqlPipeline]: 爬虫进行中...')

        # 把数据写入mysql数据库, 下载数据库包并导入：pip install pymysql
        # 确定自己的数据库中准备好了相应的数据表
        try:
            cursor = self.conn.cursor()
            # 插入的sql语句, (%s, %s, %s) 对应相应的字段类型, %s表示字符串类型
            insert_sql = 'insert into category (name, category, date) values (%s, %s, %s)'
            # execute() 的第二个参数是一个元组, 里面的每一个元素对应 sql 语句中的字段值
            cursor.execute(insert_sql, (item['name'], item['category'], item['date']))
            # 提交事务
            self.conn.commit()
        except:
            self.conn.rollback()    # 出现异常, 执行回滚操作
        finally:
            if cursor:
                cursor.close()

        return item  # 把数据传递给下一个管道

    def close_spider(self, spider):
        print('[Game4399MysqlPipeline]: 爬虫关闭了...')
        if self.conn:
            self.conn.close()

# 自定义一个管道程序, 记得在 settings.py 文件中配置, 否则不生效
class OtherPipeline:
    def process_item(self, item, spider):
        # 比如这里给传递过来的数据添加一个新的字段
        # item['new_field'] = 'hello'
        return item
