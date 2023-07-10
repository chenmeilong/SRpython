import pandas as pd
import numpy as np

# 读取 excel 文件
data = pd.read_excel('example.xlsx', sheet_name='Sheet1')
print (data)
#pd.read_excel(io, sheet_name=0, header=0, names=None, index_col=None, usecols=None)
#io：excel文件
# sheet_name：返回指定的sheet，如果将sheet_name指定为None，则返回全表，如果需要返回多个表，可以将sheet_name指定为一个列表，例如['sheet1', 'sheet2']
# header：指定数据表的表头，默认值为0，即将第一行作为表头。
# usecols：读取指定的列，例如想要读取第一列和第二列数据：
#          pd.read_excel("example.xlsx", sheet_name=None, usecols=[0, 1])


# 找到gender这一列，再在这一列中进行比较
data['gender'][data['gender'] == 'male'] = 0
data['gender'][data['gender'] == 'female'] = 1
print (data)










