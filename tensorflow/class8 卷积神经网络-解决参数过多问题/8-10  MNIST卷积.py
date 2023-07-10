# 需要在第五章的目录下运行，
import tensorflow as tf
# 导入 MINST 数据集  28*28数据集
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/data/", one_hot=True)


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')      #x =input输入图像 [一个批次图片数量，图片高度，宽度，图像通道数]  W =filter  [卷积核高度，宽度，图像通道数，录波器个数] strdes步长一维向量长度为4   返回值为feature map
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')           #输入x  teaturemap  [batch,height,width,channels]  ksize窗口大小  strides步长  返回一个tensor [batch,height,width,channels]

def avg_pool_7x7(x):
    return tf.nn.avg_pool(x, ksize=[1, 7, 7, 1],
                          strides=[1, 7, 7, 1], padding='SAME')

# tf Graph Input
x = tf.placeholder(tf.float32, [None, 784])  # mnist data维度 28*28=784
y = tf.placeholder(tf.float32, [None, 10])  # 0-9 数字=> 10 classes

W_conv1 = weight_variable([5, 5, 1, 32])     #5*5  1通道  32层
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1, 28, 28, 1])      #   -1表示自动计算

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)                                          #7*7
#########################################################new
W_conv3 = weight_variable([5, 5, 64, 10])
b_conv3 = bias_variable([10])
h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)

nt_hpool3 = avg_pool_7x7(h_conv3)  # 64
nt_hpool3_flat = tf.reshape(nt_hpool3, [-1, 10])
y_conv = tf.nn.softmax(nt_hpool3_flat)

cross_entropy = -tf.reduce_sum(y * tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# 启动session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(2000):  # 20000准确率高
        batch = mnist.train.next_batch(50)  # 50
        if i % 20 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x: batch[0], y: batch[1]})
            print("step %d, training accuracy %g" % (i, train_accuracy))
        train_step.run(feed_dict={x: batch[0], y: batch[1]})

    print("test accuracy %g" % accuracy.eval(feed_dict={
        x: mnist.test.images, y: mnist.test.labels}))

