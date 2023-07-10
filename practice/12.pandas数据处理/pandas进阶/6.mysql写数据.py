#将CSV文件写入到MySQL中
# to_sql函数并不在pd之中，而是在io.sql之中，是sql脚本下的一个类！！！所以to_sql的最好写法就是：
#         pd.io.sql.to_sql(df1,tablename,con=conn,if_exists='repalce')
# 导入必要模块
import pandas as pd
from sqlalchemy import create_engine

# 初始化数据库连接，使用pymysql模块
db_info = {'user': 'root',
           'password': '123456',
           'host': 'localhost',
           'port': 3306,
           'database': 'dht11data'
           }

engine = create_engine('mysql+pymysql://%(user)s:%(password)s@%(host)s:%(port)d/%(database)s?charset=utf8' % db_info, encoding='utf-8')
# 读取本地CSV文件
df = pd.read_csv("csv记事本.csv", sep=',')
print(df)


#DataFrame.to_sql(name, con, schema=None, if_exists='fail', index=True, index_label=None, chunksize=None, dtype=None)
# name：输出的表名
# con：连接数据库的引擎
# index：是否将DataFrame的index单独写到一列中，默认为“True”
# index_label：当index为True时，指定列作为DataFrame的index输出
# dtype：指定列的数据类型，字典形式存储{column_name: sql_dtype}，常见数据类型是sqlalchemy.types.INT()和sqlalchemy.types.CHAR(length=x)。
# 注意：INT和CHAR都需要大写，INT()不用指定长度。
# if_exists:
# 1.fail:如果表存在，啥也不做
# 2.replace:如果表存在，删了表，再建立一个新表，把数据插入
# 3.append:如果表存在，把数据插入，如果表不存在创建一个表！！
pd.io.sql.to_sql(df, 'example', con=engine, index=False, if_exists='replace')

#df.to_sql('example', con=engine,  if_exists='replace')        #这种形式也可以

print("Write to MySQL successfully!")


