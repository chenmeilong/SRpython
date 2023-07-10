import pandas  as pd

a=1
string='acc_exp'+str(a)+'_user'+str(int((a)/2))
string1='HAPT Data Set/RawData/'+string+'.txt'
string2='HAPT Data Set/sourcedata/'+string+'.csv'

csv_data = pd.read_csv(string1,sep = ' ')  # 读取训练数据
print(csv_data.shape)

a=csv_data.iloc[:,0].size#行数
a=a+1
csv_data.index=range(1, a)

csv_data.columns=['X','Y','Z']
#csv_data.loc[0] = [9, 10, 11]      #改  一行
csv_data.to_csv(string2)