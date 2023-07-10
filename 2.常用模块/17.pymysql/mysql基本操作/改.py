import pymysql

# 3.更新操作
db = pymysql.connect(host="localhost", user="root",
                     password="123456", db='mysql', port=3306)

# 使用cursor()方法获取操作游标
cur = db.cursor()

sql_update = "update user1 set name = '%s' where id = %d"

try:
    cur.execute(sql_update % ("xiongda", 4))  # 像sql语句传递参数
    # 提交
    db.commit()
except Exception as e:
    # 错误回滚
    db.rollback()
finally:
    db.close()
