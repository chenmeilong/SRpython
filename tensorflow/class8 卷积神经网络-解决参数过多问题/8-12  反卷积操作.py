#tf.nn.conv2d_transpose(value,filter,output_shape,strides,padding="SAME"....)
#value原来卷积完的张量
#filter：卷积核
#output_shape：原来卷积前的形状 5*5/4*4
#strides：步长
#padding ：SAME补齐边缘 ,VALID不补齐
import numpy as np
import tensorflow as tf 

img = tf.Variable(tf.constant(1.0,shape = [1, 4, 4, 1])) 

filter =  tf.Variable(tf.constant([1.0,0,-1,-2],shape = [2, 2, 1, 1]))

conv = tf.nn.conv2d(img, filter, strides=[1, 2, 2, 1], padding='VALID')  
cons = tf.nn.conv2d(img, filter, strides=[1, 2, 2, 1], padding='SAME')
print(conv.shape)
print(cons.shape)
 
contv= tf.nn.conv2d_transpose(conv, filter, [1,4,4,1],strides=[1, 2, 2, 1], padding='VALID')
conts = tf.nn.conv2d_transpose(cons, filter, [1,4,4,1],strides=[1, 2, 2, 1], padding='SAME')
 
with tf.Session() as sess:  
    sess.run(tf.global_variables_initializer() )  

    print("conv:\n",sess.run([conv,filter]))    #不补0卷积
    print("cons:\n",sess.run([cons]))           #补0卷积
    print("contv:\n",sess.run([contv]))         #不补0反卷积
    print("conts:\n",sess.run([conts]))         #补0反卷积
