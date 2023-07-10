import pandas as pd
import numpy as np

dates = pd.date_range('20170101', periods=6)
df = pd.DataFrame(np.random.randn(6,4), index=dates, columns=list('ABCD'))
print(df)
print("--------------"*5)

#使用单列的值来选择数据
print(df[df.A > 0])
print("--------------"*5)

#从满足布尔条件的DataFrame中选择值
print(df[df > 0])
print("--------------"*5)


#使用isin()方法进行过滤

dates = pd.date_range('20170101', periods=6)
df = pd.DataFrame(np.random.randn(6,4), index=dates, columns=list('ABCD'))

df2 = df.copy()
df2['E'] = ['one', 'one','two','three','four','three']
print(df2)
print("============= start to filter =============== ")
print(df2[df2['E'].isin(['two','four'])])





