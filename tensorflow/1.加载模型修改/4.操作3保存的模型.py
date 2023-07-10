import tensorflow as tf
#graph.get_tensor_by_name()方法来操纵这个保存的模型。
sess = tf.Session()
# 先加载图和参数变量
saver = tf.train.import_meta_graph('./checkpoint_dir/MyModel-1000.meta')
saver.restore(sess, tf.train.latest_checkpoint('./checkpoint_dir'))

# 访问placeholders变量，并且创建feed-dict来作为placeholders的新值
graph = tf.get_default_graph()
w1 = graph.get_tensor_by_name("w1:0")
w2 = graph.get_tensor_by_name("w2:0")
feed_dict = {w1: 13.0, w2: 17.0}

# 接下来，访问你想要执行的op
op_to_restore = graph.get_tensor_by_name("op_to_restore:0")

print(sess.run(op_to_restore, feed_dict))
# 打印结果为60.0==>(13+17)*2