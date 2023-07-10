#提取图片二维特征，并还原图片  线性解码器：有一个线性输出层，编码最后一层没有sigmoid交换  重点：onehot编码 转 数字 图像显示 ///输出tf量的维度
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 导入 MINST 数据集
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/data/", one_hot=True)

#参数设置
learning_rate = 0.01    
# hidden layer settings
n_hidden_1 = 256
n_hidden_2 = 64
n_hidden_3 = 16
n_hidden_4 = 2
n_input = 784  # MNIST data 输入 (img shape: 28*28)

#tf Graph输入
x = tf.placeholder("float", [None,n_input])
y=x
weights = {
    'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1],)),
    'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2],)),
    'encoder_h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3],)),
    'encoder_h4': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4],)),

    'decoder_h1': tf.Variable(tf.random_normal([n_hidden_4, n_hidden_3],)),
    'decoder_h2': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_2],)),
    'decoder_h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1],)),
    'decoder_h4': tf.Variable(tf.random_normal([n_hidden_1, n_input],)),
	} 

biases = {
    'encoder_b1': tf.Variable(tf.zeros([n_hidden_1])),
    'encoder_b2': tf.Variable(tf.zeros([n_hidden_2])),
    'encoder_b3': tf.Variable(tf.zeros([n_hidden_3])),
    'encoder_b4': tf.Variable(tf.zeros([n_hidden_4])),

    'decoder_b1': tf.Variable(tf.zeros([n_hidden_3])),
    'decoder_b2': tf.Variable(tf.zeros([n_hidden_2])),
    'decoder_b3': tf.Variable(tf.zeros([n_hidden_1])),
    'decoder_b4': tf.Variable(tf.zeros([n_input])),
	}


def encoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   biases['encoder_b2']))
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['encoder_h3']),
                                   biases['encoder_b3']))
    print(layer_3.shape)  # 16维数据
    layer_4 = tf.add(tf.matmul(layer_3, weights['encoder_h4']),            #没有sigmoid激活 所有叫线性
                                    biases['encoder_b4'])
    print(layer_4.shape)    #2维数据
    return layer_4


def decoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                   biases['decoder_b2']))
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['decoder_h3']),
                                biases['decoder_b3']))
    layer_4 = tf.nn.sigmoid(tf.add(tf.matmul(layer_3, weights['decoder_h4']),
                                biases['decoder_b4']))
    return layer_4


# 构建模型
encoder_op = encoder(x) 			
y_pred = decoder(encoder_op)	# 784 Features

cost = tf.reduce_mean(tf.pow(y - y_pred, 2))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)



#训练
training_epochs = 20	# 20 Epoch 训练
batch_size = 256
display_step = 1

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    total_batch = int(mnist.train.num_examples/batch_size)
    # 启动循环开始训练
    for epoch in range(training_epochs):
        # 遍历全部数据集
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)  
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs})
        # 显示训练中的详细信息
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1),
                  "cost=", "{:.9f}".format(c))

    print("完成!")

    # 可视化结果
    show_num = 10
    encode_decode = sess.run(
        y_pred, feed_dict={x: mnist.test.images[:show_num]})
    # 将样本对应的自编码重建图像一并输出比较
    f, a = plt.subplots(2, 10, figsize=(10, 2))
    for i in range(show_num):
        a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
        a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))
    plt.show()

    aa = [np.argmax(l) for l in mnist.test.labels]#将onehot编码转成一般编码
    encoder_result = sess.run(encoder_op, feed_dict={x: mnist.test.images})
    plt.scatter(encoder_result[:, 0], encoder_result[:, 1], c=aa)#mnist.test.labels)
    plt.colorbar()
    plt.show()

    # 换一种噪声测试一个
    randidx = np.random.randint(test_X.shape[0], size=1)
    orgvec = test_X[randidx, :]
    testvec = test_X[randidx, :]
    label = np.argmax(test_Y[randidx, :], 1)

    print("label is %d" % (label))
    # Noise type

    print("Salt and Pepper Noise")
    noisyvec = testvec
    rate = 0.15
    noiseidx = np.random.randint(test_X.shape[1]
                                 , size=int(test_X.shape[1] * rate))
    noisyvec[0, noiseidx] = 1 - noisyvec[0, noiseidx]

    outvec = sess.run(reconstruction, feed_dict={x: noisyvec, dropout_keep_prob: 1})
    outimg = np.reshape(outvec, (28, 28))

    # Plot
    plt.matshow(np.reshape(orgvec, (28, 28)), cmap=plt.get_cmap('gray'))
    plt.title("Original Image")
    plt.colorbar()

    plt.matshow(np.reshape(noisyvec, (28, 28)), cmap=plt.get_cmap('gray'))
    plt.title("Input Image")
    plt.colorbar()

    plt.matshow(outimg, cmap=plt.get_cmap('gray'))
    plt.title("Reconstructed Image")
    plt.colorbar()
    plt.show()

