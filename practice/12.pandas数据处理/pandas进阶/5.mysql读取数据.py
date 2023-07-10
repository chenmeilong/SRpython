#三个库来实现Pandas读写MySQL数据库：
# pandas、
# sqlalchemy     sqlalchemy模块实现了与不同数据库的连接
# pymysql      pymysql模块则使得Python能够操作MySQL数据库。
#pandas模块提供了read_sql_query（）函数实现了对数据库的查询，to_sql（）函数实现了对数据库的写入。

#engine.execute(sql)可以直接执行sql语句
#如果用pymysql，则必须用cursor

import pandas as pd
from sqlalchemy import create_engine

# 初始化数据库连接，使用pymysql模块
# MySQL的用户：root, 密码:123456, 端口：3306,数据库：test
engine = create_engine('mysql+pymysql://root:123456@localhost:3306/dht11data')
# 查询语句，选出user1表中的所有数据
sql = ''' select * from user1; '''
# read_sql_query的两个参数: sql语句， 数据库连接
df = pd.read_sql_query(sql, engine)                    #输出mysql数据
# 输出employee表的查询结果
print(df)

# 新建pandas中的DataFrame, 只有id,num两列
df = pd.DataFrame({'id': [1, 2, 3, 4], 'name': ['zhangsan', 'lisi', 'wangwu', 'zhuliu']})
# 将新建的DataFrame储存为MySQL中的数据表，储存index列
df.to_sql('mydf', engine, index=True)                                               # 新建表储存数据
print('Read from and write to Mysql table successfully!')













