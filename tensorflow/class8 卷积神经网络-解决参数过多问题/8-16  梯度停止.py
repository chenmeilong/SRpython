#梯度停止例子
import tensorflow as tf
tf.reset_default_graph()
w1 = tf.get_variable('w1', shape=[2])
w2 = tf.get_variable('w2', shape=[2])

w3 = tf.get_variable('w3', shape=[2])
w4 = tf.get_variable('w4', shape=[2])

y1 = w1 + w2+ w3
y2 = w3 + w4

a = w1+w2
a_stoped = tf.stop_gradient(a)    #停止梯度计算  将w1+w2置None
y3= a_stoped+w3

# grad_ys梯度的y值
gradients = tf.gradients([y1, y2], [w1, w2, w3, w4], grad_ys=[tf.convert_to_tensor([1.,2.]),
                                                          tf.convert_to_tensor([3.,4.])])
                                                          
gradients2 = tf.gradients(y3, [w1, w2, w3], grad_ys=tf.convert_to_tensor([1.,2.]))             #梯度被停止报错
print(gradients2) 
 
gradients3 = tf.gradients(y3, [ w3], grad_ys=tf.convert_to_tensor([1.,2.])) 
                                                       
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(gradients))
    #print(sess.run(gradients2))#报错
    print(sess.run(gradients3))
    