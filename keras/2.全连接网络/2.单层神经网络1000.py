#在前面 模型中的隐藏层神经元个数 改为1000  过拟合现象加剧    accuracy= 0.9797
from keras.utils import np_utils
import numpy as np
from keras.datasets import mnist
np.random.seed(10)

## # 数据导入＋预处理
(x_train_image, y_train_label), (x_test_image, y_test_label) = mnist.load_data()
x_Train = x_train_image.reshape(60000, 784).astype('float32')
x_Test = x_test_image.reshape(10000, 784).astype('float32')
x_Train_normalize = x_Train / 255
x_Test_normalize = x_Test / 255
y_Train_OneHot = np_utils.to_categorical(y_train_label)
y_Test_OneHot  = np_utils.to_categorical(y_test_label)

# # 建立模型
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(units=1000,input_dim=784,kernel_initializer='normal',activation='relu'))       #输入784  中间隐藏层有1000个神经元
model.add(Dense(units=10,kernel_initializer='normal',activation='softmax'))      #'normal'表示正太分布随机数初始化
print(model.summary())  #打印模型结构   param为需要更新的 权重和偏置值个数   728*1000+1000   256*10+10

# 定义训练方式
model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])   #评估模型方式为准确率
#开始训练
train_history = model.fit(x=x_Train_normalize, y=y_Train_OneHot,validation_split = 0.2,epochs = 10, batch_size = 200, verbose = 2)
#verbose显示训练过程  validation_split = 0.2  80%数据作为训练  20%作为验证数据集

# # 以图形显示训练过程
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
scores = model.evaluate(x_Test_normalize, y_Test_OneHot)     #model.evaluate 评估模型准确率
print('accuracy=', scores[1])    #准确率
