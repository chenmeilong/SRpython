import tensorflow as tf
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
savedir = "log/"
print_tensors_in_checkpoint_file(savedir+"linermodel.cpkt", None, True)   #输出模型内容

W = tf.Variable(1.0, name="weight")
b = tf.Variable(2.0, name="bias")

# 放到一个字典里:
saver = tf.train.Saver({'weight': b, 'bias': W})

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    saver.save(sess, savedir+"linermodel.cpkt")

print_tensors_in_checkpoint_file(savedir+"linermodel.cpkt", None, True)