import tensorflow as tf
a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)
add = tf.add(a, b)
mul = tf.multiply(a, b)                      #a与b相乘
with tf.Session() as sess:
    # Run every operation with variable input
    print ("相加: %i" % sess.run(add, feed_dict={a: 3, b: 4}))
    print ("相乘: %i" % sess.run(mul, feed_dict={a: 3, b: 4}))
    print (sess.run([add,mul], feed_dict={a: 3, b: 4}))