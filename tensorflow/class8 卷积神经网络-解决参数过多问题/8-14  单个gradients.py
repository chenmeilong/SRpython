#gradients(求导公式的结果，针对哪个变量求偏导)  #求梯度   #此代码运行不了
import tensorflow as tf

w1 = tf.Variable([[1,2]])
w2 = tf.Variable([[3,4]])

y = tf.matmul(w1, [[9],[10]])
#grads = tf.gradients(y,[w1,w2])#w2不相干，会报错
grads = tf.gradients(y,[w1])    #求w1的梯度

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    gradval = sess.run(grads)
    print(gradval)
