
#放在cifar目录下
import  cifar10_input
import tensorflow as tf
import pylab 

#取数据
batch_size = 128
data_dir = '/pythoncode/tensorflow/tmp/cifar10_data/cifar-10-batches-bin'
images_test, labels_test = cifar10_input.inputs(eval_data = True, data_dir = data_dir, batch_size = batch_size)


sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
tf.train.start_queue_runners()
image_batch, label_batch = sess.run([images_test, labels_test])
print("__\n",image_batch[0])
print("__\n",label_batch[0])
pylab.imshow(image_batch[0])
pylab.show()


# sess = tf.Session()                                     #资源随程序关闭整体销毁，结束后不会报错
# tf.global_variables_initializer().run(session=sess)
# tf.train.start_queue_runners(sess=sess)                          #队列
# image_batch, label_batch = sess.run([images_test, labels_test])
# print("__\n",image_batch[0])
# print("__\n",label_batch[0])
# pylab.imshow(image_batch[0])
# pylab.show()

# with tf.Session() as sess:               #结束后保存，原因：with的session是自动关闭的 此时队列还在写数据
#    tf.global_variables_initializer().run()
#    tf.train.start_queue_runners()
#    image_batch, label_batch = sess.run([images_test, labels_test])
#    print("__\n",image_batch[0])
#    print("__\n",label_batch[0])
#    pylab.imshow(image_batch[0])
#    pylab.show()
