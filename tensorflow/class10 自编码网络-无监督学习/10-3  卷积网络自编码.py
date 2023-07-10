#卷积网络的自编码         #需要在GPU上运行
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 导入 MINST 数据集
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/data/", one_hot=True)

#最大池化
def max_pool_with_argmax(net, stride):
    _, mask = tf.nn.max_pool_with_argmax( net,ksize=[1, stride, stride, 1], strides=[1, stride, stride, 1],padding='SAME')
    mask = tf.stop_gradient(mask)
    net = tf.nn.max_pool(net, ksize=[1, stride, stride, 1],strides=[1, stride, stride, 1], padding='SAME') 
    return net, mask
#4*4----2*2--=2*2 【6，8，12，16】    
# 反池化函数
def unpool(net, mask, stride):
    ksize = [1, stride, stride, 1]
    input_shape = net.get_shape().as_list()
    #  calculation new shape
    output_shape = (input_shape[0], input_shape[1] * ksize[1], input_shape[2] * ksize[2], input_shape[3])
    # calculation indices for batch, height, width and feature maps
    one_like_mask = tf.ones_like(mask)
    batch_range = tf.reshape(tf.range(output_shape[0], dtype=tf.int64), shape=[input_shape[0], 1, 1, 1])
    b = one_like_mask * batch_range
    y = mask // (output_shape[2] * output_shape[3])
    x = mask % (output_shape[2] * output_shape[3]) // output_shape[3]
    feature_range = tf.range(output_shape[3], dtype=tf.int64)
    f = one_like_mask * feature_range
    # transpose indices & reshape update values to one dimension
    updates_size = tf.size(net)
    indices = tf.transpose(tf.reshape(tf.stack([b, y, x, f]), [4, updates_size]))
    values = tf.reshape(net, [updates_size])
    ret = tf.scatter_nd(indices, values, output_shape)
    return ret


def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')  
                        
# 网络模型参数
learning_rate = 0.01
n_conv_1 = 16 # 第一层16个ch
n_conv_2 = 32 # 第二层32个ch
n_input = 784 # MNIST data 输入 (img shape: 28*28)
batchsize = 50

# 占位符
x = tf.placeholder("float", [batchsize, n_input])#输入

x_image = tf.reshape(x, [-1,28,28,1])


# 编码
def encoder(x):
    h_conv1 = tf.nn.relu(conv2d(x, weights['encoder_conv1']) + biases['encoder_conv1'])
    h_conv2 = tf.nn.relu(conv2d(h_conv1, weights['encoder_conv2']) + biases['encoder_conv2'])  
    return h_conv2,h_conv1

# 解码
def decoder(x,conv1):
    t_conv1 = tf.nn.conv2d_transpose(x-biases['decoder_conv2'], weights['decoder_conv2'], conv1.shape,[1,1,1,1])
    t_x_image = tf.nn.conv2d_transpose(t_conv1-biases['decoder_conv1'], weights['decoder_conv1'], x_image.shape,[1,1,1,1])
    return t_x_image


#学习参数     卷积核
weights = {
    'encoder_conv1': tf.Variable(tf.truncated_normal([5, 5, 1, n_conv_1],stddev=0.1)),
    'encoder_conv2': tf.Variable(tf.random_normal([3, 3, n_conv_1, n_conv_2],stddev=0.1)),
    'decoder_conv1': tf.Variable(tf.random_normal([5, 5, 1, n_conv_1],stddev=0.1)),
    'decoder_conv2': tf.Variable(tf.random_normal([3, 3, n_conv_1, n_conv_2],stddev=0.1))
}
biases = {
    'encoder_conv1': tf.Variable(tf.zeros([n_conv_1])),
    'encoder_conv2': tf.Variable(tf.zeros([n_conv_2])),
    'decoder_conv1': tf.Variable(tf.zeros([n_conv_1])),
    'decoder_conv2': tf.Variable(tf.zeros([n_conv_2])),
}


#输出的节点
encoder_out,conv1 = encoder(x_image)
h_pool2, mask = max_pool_with_argmax(encoder_out, 2)

h_upool = unpool(h_pool2, mask, 2)
pred = decoder(h_upool,conv1)


# 使用平方差为cost
cost = tf.reduce_mean(tf.pow(x_image - pred, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

# 训练参数
training_epochs = 20  #一共迭代20次

display_step = 5     #迭代5次输出一次信息

# 启动绘话
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    
    total_batch = int(mnist.train.num_examples/batchsize)
    # 开始训练
    for epoch in range(training_epochs):#迭代
        
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batchsize)#取数据
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs})# 训练模型
        if epoch % display_step == 0:# 现实日志信息
            print("Epoch:", '%04d' % (epoch+1),"cost=", "{:.9f}".format(c))

    print("完成!")
    
    # 测试
    batch_xs, batch_ys = mnist.train.next_batch(batchsize)
    print ("Error:", cost.eval({x: batch_xs}))

    # 可视化结果
    show_num = 10
    reconstruction = sess.run(
        #pred, feed_dict={x: mnist.test.images[:show_num]})
        pred, feed_dict={x: batch_xs})

    f, a = plt.subplots(2, 10, figsize=(10, 2))
    for i in range(show_num):
        #a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
        a[0][i].imshow(np.reshape(batch_xs[i], (28, 28)))
        a[1][i].imshow(np.reshape(reconstruction[i], (28, 28)))
    plt.draw()










































