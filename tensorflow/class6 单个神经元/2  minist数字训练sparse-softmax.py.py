#需要在第五章的目录下运行，
#在原来的基础上修改了
import tensorflow as tf #导入tensorflow库
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/")         #去掉了, one_hot=True  不需要one hot编码 参考书上107页
tf.reset_default_graph()    #重置图
# tf Graph Input
x = tf.placeholder(tf.float32, [None, 784]) # mnist data维度 28*28=784
y = tf.placeholder(tf.int32, [None]) # 0-9 数字=> 10 classes       #数据类型修改  (tf.float32, [None, 10])

# Set model weights
W = tf.Variable(tf.random_normal([784, 10]))    #随机值
b = tf.Variable(tf.zeros([10]))       #置0

z= tf.matmul(x, W) + b
# 构建模型
pred = tf.nn.softmax(z) # Softmax分类

# Minimize error using cross entropy  损失函数
#cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))
cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=z))     #函数修改
#参数设置
learning_rate = 0.01
# 使用梯度下降优化器
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

training_epochs = 25     #迭代25次
batch_size = 100         #训练过程中一次取100个数据训练
display_step = 1        #每训练一次打印中间状态
saver = tf.train.Saver()
model_path = "log/521model.ckpt"

# 启动session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())# Initializing OP   运行初始化

    # 启动循环开始训练
    for epoch in range(training_epochs):     #循环25次
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        # 遍历全部数据集
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs,
                                                          y: batch_ys})
            # Compute average loss
            avg_cost += c / total_batch
        # 显示训练中的详细信息
        if (epoch+1) % display_step == 0:
            print ("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

    print( " Finished!")