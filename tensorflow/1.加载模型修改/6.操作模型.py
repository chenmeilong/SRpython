#如果只想恢复图的一部分，并且再加入其它的op用于fine-tuning。只需通过graph.get_tensor_by_name()方法获取需要的op，并且在此基础上建立图

# ......
# ......
# saver = tf.train.import_meta_graph('vgg.meta')
# # 访问图
# graph = tf.get_default_graph()
#
# # 访问用于fine-tuning的output
# fc7 = graph.get_tensor_by_name('fc7:0')
#
# # 如果你想修改最后一层梯度，需要如下
# fc7 = tf.stop_gradient(fc7)  # It's an identity function
# fc7_shape = fc7.get_shape().as_list()
#
# new_outputs = 2
# weights = tf.Variable(tf.truncated_normal([fc7_shape[3], num_outputs], stddev=0.05))
# biases = tf.Variable(tf.constant(0.05, shape=[num_outputs]))
# output = tf.matmul(fc7, weights) + biases
# pred = tf.nn.softmax(output)
#
# # Now, you run this with fine-tuning data in sess.run()
#











