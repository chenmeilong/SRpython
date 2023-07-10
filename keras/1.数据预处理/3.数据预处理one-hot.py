# # 将images进行预处理
from keras.utils import np_utils       #label标签转为one-hot
from keras.datasets import mnist

(x_train_image, y_train_label), (x_test_image, y_test_label) = mnist.load_data()

print('x_train_image:', x_train_image.shape)
print('y_train_label:', y_train_label.shape)

x_Train = x_train_image.reshape(60000, 784).astype('float32')
x_Test = x_test_image.reshape(10000, 784).astype('float32')

print('x_train:', x_Train.shape)
print('x_test:', x_Test.shape)

x_Train_normalize = x_Train / 255
x_Test_normalize = x_Test / 255

# # one hot encode outputs
print (y_train_label[:5])   #输出前五个标签
y_TrainOneHot = np_utils.to_categorical(y_train_label)     #转成one-hot 编码
y_TestOneHot = np_utils.to_categorical(y_test_label)
print (y_TrainOneHot[:5])
