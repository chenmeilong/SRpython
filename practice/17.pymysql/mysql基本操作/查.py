# import pymysql  # 导入 pymysql
# # 打开数据库连接
# db = pymysql.connect(host="localhost", user="root",
#                      password="123456", db='alldata', port=3306)
# # 使用cursor()方法获取操作游标
# cur = db.cursor()
# # 1.查询操作
# # 编写sql 查询语句  user1 对应我的表名
# sql = "select * from user1"
# try:
#     cur.execute(sql)  # 执行sql语句
#
#     results = cur.fetchall()  # 获取查询的所有记录
#     print("id", "name", "password")
#     # 遍历结果
#     for row in results:
#         id = row[0]
#         name = row[1]
#         password = row[2]
#         print(id, name, password)
# except Exception as e:
#     raise e
# finally:
#     db.close()  # 关闭连接


import pymysql  # 导入 pymysql

# 打开数据库连接
db = pymysql.connect(host="localhost", user="root",
                     password="123456", db='alldata', port=3306)
# 使用cursor()方法获取操作游标
cur = db.cursor()
# 1.查询操作
# 编写sql 查询语句  user1 对应我的表名
sql = "select * from datatable"
try:
    cur.execute(sql)  # 执行sql语句

    results = cur.fetchall()  # 获取查询的所有记录
    # 遍历结果
    for row in results:
        dip = row[0]
        temp = row[1]
        humi = row[2]
        mq = row[3]
        qual = row[4]
        time = row[5]
        ID = row[6]
        print(dip,  temp,humi, mq,qual,time, ID)            #除了时间  导出后全是int类型
except Exception as e:
    raise e
finally:
    db.close()  # 关闭连接



