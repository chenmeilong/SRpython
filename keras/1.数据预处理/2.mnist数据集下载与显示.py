from keras.datasets import mnist
import matplotlib.pyplot as plt

(x_train_image, y_train_label), (x_test_image, y_test_label) = mnist.load_data()    #检测目录下有没有mnist数据集 没有就自动下载  C:\Users\Wayne\.keras\datasets
print('train data=', len(x_train_image))
print(' test data=', len(x_test_image))
print('x_train_image:', x_train_image.shape)
print('y_train_label:', y_train_label.shape)

def plot_image(image):
    fig = plt.gcf()
    fig.set_size_inches(2, 2)    #设置图像大小
    plt.imshow(image, cmap='binary')   # binary黑白灰度显示
    plt.show()
plot_image(x_train_image[0])      #显示图片
print (y_train_label[0])          #输出标签

def plot_images_labels_prediction(images, labels,prediction, idx, num=10):
    fig = plt.gcf()
    fig.set_size_inches(12, 14)    #设置画布大小
    if num > 25: num = 25
    for i in range(0, num):
        ax = plt.subplot(5, 5, 1 + i)
        ax.imshow(images[idx], cmap='binary')
        title = "label=" + str(labels[idx])
        if len(prediction) > 0:
            title += ",predict=" + str(prediction[idx])

        ax.set_title(title, fontsize=10)   #fontsize字体大小
        ax.set_xticks([]);
        ax.set_yticks([])
        idx += 1
    plt.show()

plot_images_labels_prediction(x_train_image, y_train_label, [], 0, 10)

print('x_test_image:', x_test_image.shape)
print('y_test_label:', y_test_label.shape)
plot_images_labels_prediction(x_test_image, y_test_label, [], 0, 10)