#单层256神经网络   寻找错误id号  混淆矩阵   导入 训练 测试 使用   存在过拟合问题    accuracy= 0.9769
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
model.add(Dense(units=256,input_dim=784,kernel_initializer='normal',activation='relu'))       #输入784  中间隐藏层有256个神经元
model.add(Dense(units=10,kernel_initializer='normal',activation='softmax'))      #'normal'表示正太分布随机数初始化
print(model.summary())  #打印模型结构   param为需要更新的 权重和偏置值个数   728*256+256   256*10+10

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
print('accuracy=', scores[1])    #准确率     scores[0]  是loss

# # 进行预测
prediction = model.predict_classes(x_Test)
print (prediction)      #输出预测结果

def plot_images_labels_prediction(images, labels, prediction,idx, num=10):
    fig = plt.gcf()
    fig.set_size_inches(12, 14)
    if num > 25: num = 25
    for i in range(0, num):
        ax = plt.subplot(5, 5, 1 + i)
        ax.imshow(images[idx], cmap='binary')
        title = "label=" + str(labels[idx])
        if len(prediction) > 0:
            title += ",predict=" + str(prediction[idx])

        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        idx += 1
    plt.show()
plot_images_labels_prediction(x_test_image, y_test_label, prediction, idx=0)

# # confusion matrix
import pandas as pd
cross=pd.crosstab(y_test_label, prediction,rownames=['label'], colnames=['predict'])   #建立混淆矩阵
print (cross)   #输出混淆矩阵  查看混淆度

df = pd.DataFrame({'label': y_test_label, 'predict': prediction})
print (df[:2])
a=df[(df.label == 5) & (df.predict == 3)]   #这里寻找预测出错的id号

plot_images_labels_prediction(x_test_image, y_test_label , prediction, idx=340, num=1)
plot_images_labels_prediction(x_test_image, y_test_label , prediction, idx=1289, num=1)
