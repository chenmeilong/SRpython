import tensorflow as tf
tf.reset_default_graph()
global_step = tf.train.get_or_create_global_step()
step = tf.assign_add(global_step, 1)

with tf.train.MonitoredTrainingSession(checkpoint_dir='log/checkpoints',save_checkpoint_secs  = 2) as sess:#save_checkpoint_secs  = 2s保存一次   ，默认为10分钟
    print(sess.run([global_step]))
    while not sess.should_stop():
        i = sess.run( step)
        print( i)
