from tensorflow.examples.tutorials.mnist import input_data        #理论上可以下载好，实际会出问题，需要手动下载4个压缩包到MNIST_data目录下
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)   #onehost将本标签转换成onehost编码     下载网址：http://yann.lecun.com/exdb/mnist/

print ('输入数据:',mnist.train.images)
print ('输入数据打印shape:',mnist.train.images.shape)        #训练数据集

import pylab 
im = mnist.train.images[1]
im = im.reshape(-1,28)
pylab.imshow(im)
pylab.show()


print ('输入数据打印shape:',mnist.test.images.shape)         #测试数据集
print ('输入数据打印shape:',mnist.validation.images.shape)    #测试数据集














