#创建链接的基本使用
# 导入pymysql模块
import pymysql

# 连接database
conn = pymysql.connect(host="localhost", user="root",
                     password="123456",db='alldata', port=3306)

# 得到一个可以执行SQL语句的光标对象
cursor = conn.cursor()  # 执行完毕返回的结果集默认以元组显示
# 得到一个可以执行SQL语句并且将结果作为字典返回的游标
# cursor = conn.cursor(cursor=pymysql.cursors.DictCursor)

# 定义要执行的SQL语句                 #设置id号  int和主键     name char10   age tinyint
sql = """
CREATE TABLE USER1 (
id INT auto_increment PRIMARY KEY ,
name CHAR(10) NOT NULL UNIQUE,
age TINYINT NOT NULL
)ENGINE=innodb DEFAULT CHARSET=utf8;       
"""                                                    #注意：charset='utf8' 不能写成utf-8

# 执行SQL语句
cursor.execute(sql)

# 关闭光标对象
cursor.close()

# 关闭数据库连接
conn.close()