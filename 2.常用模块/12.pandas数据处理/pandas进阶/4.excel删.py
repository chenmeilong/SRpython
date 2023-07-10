
import pandas as pd

data = pd.read_excel("saveexample.xlsx", sheet_name='Sheet1')
print(data)
# 删除gender列，需要指定axis为1，当删除行时，axis为0
data = data.drop('gender', axis=1)
# 删除第3,4行，这里下表以0开始，并且标题行不算在类
data = data.drop([2, 3], axis=0)
print(data)

data.to_excel('saveexample.xlsx', sheet_name='Sheet1', index=False, header=True)
