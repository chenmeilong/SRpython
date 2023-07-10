import tensorflow as tf
a = tf.constant(3)                     #定义常量3
b = tf.constant(4)                     #定义常量4
with tf.Session() as sess:           #建立session
    print ("相加: %i" % sess.run(a+b))
    print( "相乘: %i" % sess.run(a*b))
