

import tensorflow as tf
img1 = tf.constant(value=[[[[1],[2],[3],[4]],[[1],[2],[3],[4]],[[1],[2],[3],[4]],[[1],[2],[3],[4]]]],dtype=tf.float32)
img2 = tf.constant(value=[[[[1],[1],[1],[1]],[[1],[1],[1],[1]],[[1],[1],[1],[1]],[[1],[1],[1],[1]]]],dtype=tf.float32)
img = tf.concat(values=[img1,img2],axis=3)
sess=tf.Session()
#sess.run(tf.initialize_all_variables())
sess.run(tf.global_variables_initializer())
print("out1=",type(img))
#转化为numpy数组
img_numpy=img.eval(session=sess)             #  tf.Tensor.eval()   当默认的会话被指定之后可以通过其计算一个张量的取值。
print("out2=",type(img_numpy))
#转化为tensor
img_tensor= tf.convert_to_tensor(img_numpy)
print("out2=",type(img_tensor))







