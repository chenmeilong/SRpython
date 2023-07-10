# import pymysql     #最好不要写id号    名不能重复   注意看创建表的格式 输入要求
# # 2.插入操作
# db = pymysql.connect(host="localhost", user="root",
#                      password="123456",  db='alldata', port=3306)
# # 使用cursor()方法获取操作游标
# cur = db.cursor()
# sql_insert = """insert into user1(name,age) values('zhu',7)"""
# try:
#     cur.execute(sql_insert)
#     # 提交
#     db.commit()
# except Exception as e:
#     # 错误回滚
#     db.rollback()
# finally:
#     db.close()


import pymysql     #最好不要写id号    名不能重复   注意看创建表的格式 输入要求
# 2.插入操作
db = pymysql.connect(host="localhost", user="root",
                     password="123456",  db='alldata', port=3306)
# 使用cursor()方法获取操作游标
cur = db.cursor()
sql_insert = """insert into datatable (dip,temp,humi,mq,time,qual) values(128,38,34,44,'54',48)"""
try:
    cur.execute(sql_insert)
    # 提交
    db.commit()
except Exception as e:             #mysql  执行出错后回滚
    # 错误回滚
    db.rollback()
finally:
    db.close()


