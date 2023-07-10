#数据输出形式   转置 排序
import pandas as pd
import numpy as np
#查看框架的顶部和底部的数据行。
dates = pd.date_range('20170101', periods=7)
df = pd.DataFrame(np.random.randn(7,4), index=dates, columns=list('ABCD'))
print(df.head())
print("--------------")
print(df.tail(3))
print("--------------"*5)

#显示索引，列和底层numpy数据，
dates = pd.date_range('20170101', periods=7)
df = pd.DataFrame(np.random.randn(7,4), index=dates, columns=list('ABCD'))
print("index is :" )
print(df.index)                   #显示索引
print("columns is :" )
print(df.columns)                #列名
print("values is :" )
print(df.values)                #主体数据
print("--------------"*5)

#描述显示数据的快速统计摘要
dates = pd.date_range('20170101', periods=7)
df = pd.DataFrame(np.random.randn(7,4), index=dates, columns=list('ABCD'))
print(df.describe())
print("--------------"*5)

#输出数据转置
dates = pd.date_range('20170101', periods=6)
df = pd.DataFrame(np.random.randn(6,4), index=dates, columns=list('ABCD'))
print(df.T)
print("--------------"*5)

#通过轴排序
dates = pd.date_range('20170101', periods=6)
df = pd.DataFrame(np.random.randn(6,4), index=dates, columns=list('ABCD'))
print(df.sort_index(axis=1, ascending=False))      #由大到小
print("--------------"*5)

#按值排序
dates = pd.date_range('20170101', periods=6)
df = pd.DataFrame(np.random.randn(6,4), index=dates, columns=list('ABCD'))
print(df.sort_values(by='B'))
print("--------------"*5)
