import tensorflow as tf

w1 = tf.placeholder("float", name="w1")
w2 = tf.placeholder("float", name="w2")
b1 = tf.Variable(2.0, name="bias")

# 定义一个op，用于后面恢复
w3 = tf.add(w1, w2)
w4 = tf.multiply(w3, b1, name="op_to_restore")
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 创建一个Saver对象，用于保存所有变量
saver = tf.train.Saver()

# 通过传入数据，执行op
print(sess.run(w4, feed_dict={w1: 4, w2: 8}))
# 打印 24.0 ==>(w1+w2)*b1

# 现在保存模型
saver.save(sess, './checkpoint_dir/MyModel', global_step=1000)

