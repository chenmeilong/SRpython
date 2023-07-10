
import pandas as pd
import numpy as np

#通过传递的整数的位置选择，参考以下示例程序 -
dates = pd.date_range('20170101', periods=6)
df = pd.DataFrame(np.random.randn(6,4), index=dates, columns=list('ABCD'))
print(df)
print(df.iloc[3])   #第四行
print("--------------"*5)

#通过整数切片，类似于numpy/python
print(df.iloc[3:5,0:2])     #行     列
print("--------------"*5)

#通过整数位置的列表，类似于numpy/python样式
print(df.iloc[[1,2,4],[0,2]])
print("--------------"*5)

#明确切片行，
print(df.iloc[1:3,:])
print("--------------"*5)

#明确切片列，
print(df.iloc[:,1:3])
print("--------------"*5)

#要明确获取值
print(df.iloc[1,1])
print("--------------"*5)

#要快速访问标量(等同于先前的方法)
print(df.iat[1,1])
print("--------------"*5)


