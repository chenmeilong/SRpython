import tensorflow as tf

with tf.Session() as sess:
    saver = tf.train.import_meta_graph('./checkpoint_dir/MyModel-1000.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./checkpoint_dir'))
    print(sess.run('w1:0'))         #注意w1:0是tensor的name。
##Model has been restored. Above statement will print the saved value




