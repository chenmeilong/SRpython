#pandas 就是 python的excel
#
#Pandas处理以下三个数据结构 -
    # 系列(Series)    1维    系列是具有均匀数据的一维数组结构
    # 数据帧(DataFrame)  2维    常用   数据帧(DataFrame)是一个具有异构数据的二维数组。     #异构数据、大小可变、数据可变
    # 面板(Panel)      3维    面板是具有异构数据的三维数据结构。在图形表示中很难表示面板。但是一个面板可以说明为DataFrame的容器。

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
print("Hello, Pandas")

#创建系列对象
s = pd.Series([1,3,5,np.nan,6,8])           #   np.nan  未定义的变量
print(s)
print("-------------------")

#通过传递numpy数组，使用datetime索引和标记列来创建DataFrame
dates = pd.date_range('20170101', periods=7)    #创建日期循环7个       periods周期
print(dates)
df = pd.DataFrame(np.random.randn(7,4), index=dates, columns=list('ABCD'))
print(df)
print("-------------------")

#通过传递可以转换为类似系列的对象的字典来创建DataFrame。
df2 = pd.DataFrame({ 'A' : 1.,
                     'B' : pd.Timestamp('20170102'),
                     'C' : pd.Series(1,index=list(range(4)),dtype='float32'),
                     'D' : np.array([3] * 4,dtype='int32'),
                     'E' : pd.Categorical(["test","train","test","train"]),
                     'F' : 'foo' })

print(df2)          #后面可以跟这些属性
# df2.A                  df2.bool
# df2.abs                df2.boxplot
# df2.add                df2.C
# df2.add_prefix         df2.clip
# df2.add_suffix         df2.clip_lower
# df2.align              df2.clip_upper
# df2.all                df2.columns
# df2.any                df2.combine
# df2.append             df2.combine_first
# df2.apply              df2.compound
# df2.applymap           df2.consolidate
# df2.D
print("-------------------")










