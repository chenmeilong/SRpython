

import pandas as pd
import numpy as np

#选择一列，产生一个系列
dates = pd.date_range('20170101', periods=6)
df = pd.DataFrame(np.random.randn(6,4), index=dates, columns=list('ABCD'))
print(df['A'])    #只输出a列
print("--------------"*5)

#选择通过[]操作符，选择切片行。
dates = pd.date_range('20170101', periods=6)
df = pd.DataFrame(np.random.randn(6,4), index=dates, columns=list('ABCD'))
print(df[0:3])
print("========= 指定选择日期 ========")
print(df['20170102':'20170103'])
print("--------------"*5)


#使用标签获取横截面，
dates = pd.date_range('20170101', periods=6)
df = pd.DataFrame(np.random.randn(6,4), index=dates, columns=list('ABCD'))
print(df.loc[dates[0]])
print("--------------"*5)

#通过标签选择多轴
dates = pd.date_range('20170101', periods=6)
df = pd.DataFrame(np.random.randn(6,4), index=dates, columns=list('ABCD'))
print(df.loc[:,['A','B']])
print("--------------"*5)

#显示标签切片，包括两个端点
dates = pd.date_range('20170101', periods=6)
df = pd.DataFrame(np.random.randn(6,4), index=dates, columns=list('ABCD'))
print(df.loc['20170102':'20170104',['A','B']])
print("--------------"*5)

#减少返回对象的尺寸(大小)
dates = pd.date_range('20170101', periods=6)
df = pd.DataFrame(np.random.randn(6,4), index=dates, columns=list('ABCD'))
print(df.loc['20170102',['A','B']])
print("--------------"*5)

#获得标量值
dates = pd.date_range('20170101', periods=6)
df = pd.DataFrame(np.random.randn(6,4), index=dates, columns=list('ABCD'))
print(df.loc[dates[0],'A'])
print("--------------"*5)

#快速访问标量(等同于先前的方法)，
dates = pd.date_range('20170101', periods=6)
df = pd.DataFrame(np.random.randn(6,4), index=dates, columns=list('ABCD'))
print(df.at[dates[0],'A'])









