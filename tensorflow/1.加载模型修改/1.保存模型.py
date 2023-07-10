# import tensorflow as tf
#
# w1 = tf.Variable(tf.random_normal(shape=[2]), name='w1')
# w2 = tf.Variable(tf.random_normal(shape=[5]), name='w2')
# saver = tf.train.Saver()
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
# saver.save(sess, './checkpoint_dir/MyModel')

#tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=2)
#如果你希望每2小时保存一次模型，并且只保存最近的5个模型文件


import tensorflow as tf

w1 = tf.Variable(tf.random_normal(shape=[2]), name='w1')
w2 = tf.Variable(tf.random_normal(shape=[5]), name='w2')
saver = tf.train.Saver([w1, w2])
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver.save(sess, './checkpoint_dir/MyModel', global_step=1000)