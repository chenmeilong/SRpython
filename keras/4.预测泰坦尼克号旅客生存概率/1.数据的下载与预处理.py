
# # 下载泰坦尼克号上旅客的数据集
import urllib.request
import os

url = "http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic3.xls"
filepath = "data/titanic3.xls"
if not os.path.isfile(filepath):
    result = urllib.request.urlretrieve(url, filepath)      #下载文件
    print('downloaded:', result)

# 使用Pandas dataframe读取数据并进行处理
import numpy
import pandas as pd
all_df = pd.read_excel(filepath)      #读取excel文件

cols = ['survived', 'name', 'pclass', 'sex', 'age', 'sibsp',
        'parch', 'fare', 'embarked']
all_df = all_df[cols]
print (all_df[:2])        #提取出需要的信息
print (all_df.isnull().sum())     #将空的  求和
df = all_df.drop(['name'], axis=1)   #删除字段

age_mean = df['age'].mean()
df['age'] = df['age'].fillna(age_mean)    #将空 填为平均值
fare_mean = df['fare'].mean()
df['fare'] = df['fare'].fillna(fare_mean)   #将空 填为平均值
df['sex'] = df['sex'].map({'female': 0, 'male': 1}).astype(int)     #性别替换数字  0男 1女
x_OneHot_df = pd.get_dummies(data=df, columns=["embarked"])  #港口 替换成3为onehot编码
print (x_OneHot_df[:2])

# # 转换为array
ndarray = x_OneHot_df.values     #
print (ndarray.shape)            #(1309, 10)
Label = ndarray[:, 0]     #标签
Features = ndarray[:, 1:]  #特征
print (Features.shape)   #(1309, 9)
print (Features[:2])
print (Label.shape)        #(1309,)

# # 将array进行标准化
from sklearn import preprocessing
minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 1))   #数据标准化
scaledFeatures = minmax_scale.fit_transform(Features)
print (scaledFeatures[:2])
print (Label[:5])

# # 将数据分为训练数据与测试数据
msk = numpy.random.rand(len(all_df)) < 0.8
train_df = all_df[msk]
test_df = all_df[~msk]
print('total:', len(all_df),'train:', len(train_df), 'test:', len(test_df))

def PreprocessData(raw_df):
    df = raw_df.drop(['name'], axis=1)
    age_mean = df['age'].mean()
    df['age'] = df['age'].fillna(age_mean)
    fare_mean = df['fare'].mean()
    df['fare'] = df['fare'].fillna(fare_mean)
    df['sex'] = df['sex'].map({'female': 0, 'male': 1}).astype(int)
    x_OneHot_df = pd.get_dummies(data=df, columns=["embarked"])

    ndarray = x_OneHot_df.values
    Features = ndarray[:, 1:]
    Label = ndarray[:, 0]

    minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 1))   #标准化
    scaledFeatures = minmax_scale.fit_transform(Features)

    return scaledFeatures, Label

train_Features, train_Label = PreprocessData(train_df)
test_Features, test_Label = PreprocessData(test_df)

train_Features[:2]

train_Label[:2]

