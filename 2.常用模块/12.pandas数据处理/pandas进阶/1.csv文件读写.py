# CSV文件最早用在简单的数据库里，由于其格式简单，并具备很强的开放性，所以起初被扫图家用作自己图集的标记。
# CSV文件是个纯文本文件，每一行表示一张图片的许多属性。
# 你在收一套图集时，只要能找到它的CSV文件，用专用的软件校验后，你对该图集的状况就可以了如指掌。
# 每行相当于一条记录，是用“，”分割字段的纯文本数据库文件。
# 在windows系统环境上.csv文件打开方式有多种，如记事本、excel、Notepad++等，只要是文本编辑器都能正确打开。

#csv文件的读取与保存
import pandas as pd
import numpy as np

csv_data = pd.read_csv('csv记事本.txt')  # 读取训练数据
print(csv_data.shape)
csv_batch_data = csv_data.tail(2)  # 取后2条数据
print(csv_batch_data)
print("-------------------")

#保存到csv文件
dates = pd.date_range('20170101', periods=7)    #创建日期循环7个       periods周期
df = pd.DataFrame(np.random.randn(7,4), index=dates, columns=list('ABCD'))
print(df)
df.to_csv('test.csv')              #保存到csv文件
#df.to_csv("test.csv", header=False, index=False)              #隐藏行和列的  头文件


#  txt  按空格转csv文件
# import pandas as pd
# csv_data = pd.read_csv('X_test.txt',sep = ' ')  # 读取训练数据
# print(csv_data.shape)
# csv_data.to_csv('X_test.csv')










