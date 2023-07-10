#acc:  0.8109090913425793   双层神经网络
import numpy
import pandas as pd
from sklearn import preprocessing   #将array进行标准化
numpy.random.seed(10)

# # 数据准备 预处理
all_df = pd.read_excel("data/titanic3.xls")
cols = ['survived', 'name', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
all_df = all_df[cols]
msk = numpy.random.rand(len(all_df)) < 0.8
train_df = all_df[msk]
test_df = all_df[~msk]
print('total:', len(all_df), 'train:', len(train_df), 'test:', len(test_df))
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
    minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 1))
    scaledFeatures = minmax_scale.fit_transform(Features)
    return scaledFeatures, Label

train_Features, train_Label = PreprocessData(train_df)
test_Features, test_Label = PreprocessData(test_df)

# 创建模型
from keras.models import Sequential
from keras.layers import Dense, Dropout

model = Sequential()
model.add(Dense(units=40, input_dim=9,kernel_initializer='uniform',activation='relu'))
model.add(Dense(units=30,kernel_initializer='uniform', activation='relu'))
model.add(Dense(units=1,kernel_initializer='uniform',activation='sigmoid'))

# 训练模型
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
train_history = model.fit(x=train_Features, y=train_Label,validation_split=0.1,epochs=30, batch_size=30, verbose=2)
#validation_split=0.1    90%数据作为训练  10%作为验证数据集

# # 6. Print History
import matplotlib.pyplot as plt
def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
show_train_history(train_history, 'acc', 'val_acc')
show_train_history(train_history, 'loss', 'val_loss')

# # 评估模型的准确率
scores = model.evaluate(x=test_Features,y=test_Label)
print ('acc: ',scores[1])

# # 预测数据
# # 加入Jack & Rose数据
Jack = pd.Series([0, 'Jack', 3, 'male', 23, 1, 0, 5.0000, 'S'])
Rose = pd.Series([1, 'Rose', 1, 'female', 20, 1, 0, 100.0000, 'S'])
JR_df = pd.DataFrame([list(Jack), list(Rose)],
                     columns=['survived', 'name', 'pclass', 'sex','age', 'sibsp', 'parch', 'fare', 'embarked'])
all_df = pd.concat([all_df, JR_df])    #加入到总数据库中

# # 进行预测
all_Features, Label = PreprocessData(all_df)    #预处理
all_probability = model.predict(all_Features)   #预测

print (all_probability[:10])  #输出前10人的生存概率

pd = all_df
pd.insert(len(all_df.columns), 'probability', all_probability)    #插入一列

# # 预测Jack & Rose数据的生存几率
print (pd[-2:])

# # 查看生存几率高，却没有存活
a=pd[(pd['survived'] == 0) & (pd['probability'] > 0.9)]
print (a)
