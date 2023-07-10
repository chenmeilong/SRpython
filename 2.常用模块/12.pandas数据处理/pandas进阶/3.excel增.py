

import pandas as pd

data = pd.read_excel("example.xlsx", sheet_name='Sheet1')

# 增加行数据，在第5行新增
data.loc[5] = ['James', 32, 'male']
# 增加列数据，给定默认值None
data['profession'] = None

# 保存数据
data.to_excel('saveexample.xlsx', sheet_name='Sheet1', index=False, header=True)