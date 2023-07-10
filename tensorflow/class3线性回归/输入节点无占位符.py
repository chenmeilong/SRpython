
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 生成模拟数据
train_X = np.float32(np.linspace(-1, 1, 100))
train_Y = 2 * train_X + np.random.randn(*train_X.shape) * 0.3  # y=2x，但是加入了噪声
# 图形显示
plt.plot(train_X, train_Y, 'ro', label='Original data')
plt.legend()
plt.show()

# 创建模型

# 模型参数
W = tf.Variable(tf.random_normal([1]), name="weight")
b = tf.Variable(tf.zeros([1]), name="bias")
# 前向结构
z = tf.multiply(W, train_X) + b

# 反向优化
cost = tf.reduce_mean(tf.square(train_Y - z))
learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)  # Gradient descent

# 初始化变量
init = tf.global_variables_initializer()
# 参数设置
training_epochs = 20
display_step = 2

# 启动session
with tf.Session() as sess:
    sess.run(init)

    # Fit all training data
    for epoch in range(training_epochs):
        for (x, y) in zip(train_X, train_Y):    #zip函数可以接受一系列的可迭代对象作为参数，将对象中对应的元素打包成一个个tuple(元组)，然后由这些tuple(元组)组成一个list(列表)，故其返回值为list(列表)
            sess.run(optimizer)

        # 显示训练中的详细信息
        if epoch % display_step == 0:
            loss = sess.run(cost)
            print("Epoch:", epoch + 1, "cost=", loss, "W=", sess.run(W), "b=", sess.run(b))

    print(" Finished!")
    print("cost=", sess.run(cost), "W=", sess.run(W), "b=", sess.run(b))



