#卷积 cifar-10       #0.7248        # 模型的加载与保存
#加载权重
# model.load_weights("SaveModel/cifarCnnModel.h5")    #加载权重  加载权重位置在 model.compile 前面
# model.save_weights("SaveModel/cifarCnnModel.h5")    #保存权重
#保存模型和权重
#model = load_model('keras_oneline.h5')
# model.save('my_model.h5')
# model = load_model('my_model.h5')

from keras.datasets import cifar10
import numpy as np
np.random.seed(10)

# # 数据准备
(x_img_train, y_label_train), (x_img_test, y_label_test) = cifar10.load_data()
print("train data:", 'images:', x_img_train.shape," labels:", y_label_train.shape)
print("test  data:", 'images:', x_img_test.shape," labels:", y_label_test.shape)
x_img_train_normalize = x_img_train.astype('float32') / 255.0
x_img_test_normalize = x_img_test.astype('float32') / 255.0
from keras.utils import np_utils
y_label_train_OneHot = np_utils.to_categorical(y_label_train)
y_label_test_OneHot = np_utils.to_categorical(y_label_test)

# # 建立模型
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
model = Sequential()
#卷积层1与卷积层2                  注意这里的卷积了两次
model.add(Conv2D(filters=32,kernel_size=(3, 3),input_shape=(32, 32,3),activation='relu', padding='same'))
model.add(Dropout(0.3))
model.add(Conv2D(filters=32, kernel_size=(3, 3),activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#卷积层2与池化层2
model.add(Conv2D(filters=64, kernel_size=(3, 3),activation='relu', padding='same'))
model.add(Dropout(0.3))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#卷积层3与池化层3
model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(Dropout(0.3))
model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#建立神经网络(平坦层、隐藏层、输出层)
model.add(Flatten())
model.add(Dropout(0.3))
model.add(Dense(2500, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1500, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(10, activation='softmax'))
print(model.summary())

#加载权重
try:
    model.load_weights("SaveModel/cifarCnnModel.h5")
    print("加载模型成功!继续训练模型")
except :
    print("加载模型失败!开始训练一个新模型")

# # 训练模型
model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
train_history = model.fit(x_img_train_normalize, y_label_train_OneHot,validation_split=0.2,
                          epochs=2, batch_size=128, verbose=1)
#画准确率曲线
import matplotlib.pyplot as plt
def show_train_history(train_acc, test_acc):
    plt.plot(train_history.history[train_acc])
    plt.plot(train_history.history[test_acc])
    plt.title('Train History')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
show_train_history('acc', 'val_acc')
show_train_history('loss', 'val_loss')   #画损失曲线

# # 评估模型的准确率
scores = model.evaluate(x_img_test_normalize,y_label_test_OneHot, verbose=0)
print (scores[1])

# # 进行预测
prediction = model.predict_classes(x_img_test_normalize)
# # 查看预测结果
label_dict = {0: "airplane", 1: "automobile", 2: "bird", 3: "cat", 4: "deer",
              5: "dog", 6: "frog", 7: "horse", 8: "ship", 9: "truck"}
def plot_images_labels_prediction(images, labels, prediction,idx, num=10):
    fig = plt.gcf()
    fig.set_size_inches(12, 14)
    if num > 25: num = 25
    for i in range(0, num):
        ax = plt.subplot(5, 5, 1 + i)
        ax.imshow(images[idx], cmap='binary')

        title = str(i) + ',' + label_dict[labels[i][0]]
        if len(prediction) > 0:
            title += '=>' + label_dict[prediction[i]]
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]);
        ax.set_yticks([])
        idx += 1
    plt.show()
plot_images_labels_prediction(x_img_test, y_label_test,prediction, 0, 10)

# # 查看预测概率
Predicted_Probability = model.predict(x_img_test_normalize)
def show_Predicted_Probability(y, prediction,x_img, Predicted_Probability, i):
    print('label:', label_dict[y[i][0]],'predict:', label_dict[prediction[i]])  #正确值的类别，预测的类别
    plt.figure(figsize=(2, 2))
    plt.imshow(np.reshape(x_img_test[i], (32, 32, 3)))
    plt.show()
    for j in range(10):
        print(label_dict[j] +' Probability:%1.9f' % (Predicted_Probability[i][j]))    #打印预测值的概率

show_Predicted_Probability(y_label_test, prediction,x_img_test, Predicted_Probability, 0)
show_Predicted_Probability(y_label_test, prediction,x_img_test, Predicted_Probability, 3)

# # 混淆矩阵
print (prediction.shape)
print (y_label_test.shape)
print (y_label_test.reshape(-1).shape)
import pandas as pd
cross=pd.crosstab(y_label_test.reshape(-1), prediction, rownames=['label'], colnames=['predict'])
print (cross)

model.save_weights("SaveModel/cifarCnnModel.h5")
print("Saved model to disk")
