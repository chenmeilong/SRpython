#infoGAN 与ACGAN 带有辅助信息的GAN学习mnist数据特征 生成以假乱真的mnist数据模拟样本  并发现内部的潜在特征信息  P434
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.stats import norm
import tensorflow.contrib.slim as slim

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/data/")#, one_hot=True)



tf.reset_default_graph()
#生成器函数
def generator(x):
    reuse = len([t for t in tf.global_variables() if t.name.startswith('generator')]) > 0
    #print (x.get_shape())
    with tf.variable_scope('generator', reuse = reuse):
        x = slim.fully_connected(x, 1024)                          #全连接
        #print( x)
        x = slim.batch_norm(x, activation_fn=tf.nn.relu)      #批量归一池化
        x = slim.fully_connected(x, 7*7*128)
        x = slim.batch_norm(x, activation_fn=tf.nn.relu)
        x = tf.reshape(x, [-1, 7, 7, 128])
        # print '22',tensor.get_shape()
        x = slim.conv2d_transpose(x, 64, kernel_size=[4,4], stride=2, activation_fn = None)        #反卷积
        #print ('gen',x.get_shape())
        x = slim.batch_norm(x, activation_fn = tf.nn.relu)
        z = slim.conv2d_transpose(x, 1, kernel_size=[4, 4], stride=2, activation_fn=tf.nn.sigmoid)
        #print ('genz',z.get_shape())
    return z

def leaky_relu(x):
     return tf.where(tf.greater(x, 0), x, 0.01 * x)

def discriminator(x, num_classes=10, num_cont=2):
    reuse = len([t for t in tf.global_variables() if t.name.startswith('discriminator')]) > 0
    #print (reuse)
    #print (x.get_shape())
    with tf.variable_scope('discriminator', reuse=reuse):
        x = tf.reshape(x, shape=[-1, 28, 28, 1])
        x = slim.conv2d(x, num_outputs = 64, kernel_size=[4,4], stride=2, activation_fn=leaky_relu)    #卷积
        x = slim.conv2d(x, num_outputs=128, kernel_size=[4,4], stride=2, activation_fn=leaky_relu)
        #print ("conv2d",x.get_shape())
        x = slim.flatten(x)
        shared_tensor = slim.fully_connected(x, num_outputs=1024, activation_fn = leaky_relu)           #全连接
        recog_shared = slim.fully_connected(shared_tensor, num_outputs=128, activation_fn = leaky_relu)
        disc = slim.fully_connected(shared_tensor, num_outputs=1, activation_fn=None)
        disc = tf.squeeze(disc, -1)                       #删掉维度为1的
        #print ("disc",disc.get_shape())#0 or 1
        recog_cat = slim.fully_connected(recog_shared, num_outputs=num_classes, activation_fn=None)     #全连接   10类 未分类  判别类别
        recog_cont = slim.fully_connected(recog_shared, num_outputs=num_cont, activation_fn=tf.nn.sigmoid)  #全连接2分类   判别info
    return disc, recog_cat, recog_cont


batch_size = 10   # 获取样本的批次大小32
classes_dim = 10  # 10 classes
con_dim = 2  # total continuous factor隐含信息维度
rand_dim = 38           #噪声维度
n_input  = 784


x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.int32, [None])

z_con = tf.random_normal((batch_size, con_dim))#2列
z_rand = tf.random_normal((batch_size, rand_dim))#38列
z = tf.concat(axis=1, values=[tf.one_hot(y, depth = classes_dim), z_con, z_rand])#50列
gen = generator(z)
genout= tf.squeeze(gen, -1)


# labels for discriminator  判别标准结果
y_real = tf.ones(batch_size) #真
y_fake = tf.zeros(batch_size)#假

# 判别器
disc_real, class_real, _ = discriminator(x) #真样本的输出
disc_fake, class_fake, con_fake = discriminator(gen)  #模拟样本输出
pred_class = tf.argmax(class_fake, dimension=1)

# 判别器 loss
loss_d_r = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_real, labels=y_real))
loss_d_f = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake, labels=y_fake))
loss_d = (loss_d_r + loss_d_f) / 2    #判别器的loss
#print ('loss_d', loss_d.get_shape())
# generator loss
loss_g = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake, labels=y_real))     #生成器的loss
# categorical factor loss
loss_cf = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=class_fake, labels=y))#分类正确  但是样本错了
loss_cr = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=class_real, labels=y))#样本与分类都正确，但是与输入的分类 对不上
loss_c =(loss_cf + loss_cr) / 2
# continuous factor loss
loss_con =tf.reduce_mean(tf.square(con_fake-z_con))   #隐含信息变量loss

# 获得各个网络中各自的训练参数
t_vars = tf.trainable_variables()
d_vars = [var for var in t_vars if 'discriminator' in var.name]
g_vars = [var for var in t_vars if 'generator' in var.name]


disc_global_step = tf.Variable(0, trainable=False)
gen_global_step = tf.Variable(0, trainable=False)

train_disc = tf.train.AdamOptimizer(0.0001).minimize(loss_d + loss_c + loss_con, var_list = d_vars, global_step = disc_global_step)
train_gen = tf.train.AdamOptimizer(0.001).minimize(loss_g + loss_c + loss_con, var_list = g_vars, global_step = gen_global_step)



training_epochs = 3
display_step = 1



with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)

        # 遍历全部数据集
        for i in range(total_batch):

            batch_xs, batch_ys = mnist.train.next_batch(batch_size)#取数据
            feeds = {x: batch_xs, y: batch_ys}

            # Fit training using batch data
            l_disc, _, l_d_step = sess.run([loss_d, train_disc, disc_global_step],feeds)
            l_gen, _, l_g_step = sess.run([loss_g, train_gen, gen_global_step],feeds)

        # 显示训练中的详细信息
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f} ".format(l_disc),l_gen)

    print("完成!")
    
    # 测试
    print ("Result:", loss_d.eval({x: mnist.test.images[:batch_size],y:mnist.test.labels[:batch_size]})
                        , loss_g.eval({x: mnist.test.images[:batch_size],y:mnist.test.labels[:batch_size]}))
    
    # 根据图片模拟生成图片
    show_num = 10
    gensimple,d_class,inputx,inputy,con_out = sess.run(
        [genout,pred_class,x,y,con_fake], feed_dict={x: mnist.test.images[:batch_size],y: mnist.test.labels[:batch_size]})

    f, a = plt.subplots(2, 10, figsize=(10, 2))
    for i in range(show_num):
        a[0][i].imshow(np.reshape(inputx[i], (28, 28)))
        a[1][i].imshow(np.reshape(gensimple[i], (28, 28)))
        print("d_class",d_class[i],"inputy",inputy[i],"con_out",con_out[i])
        
    plt.draw()
    plt.show()  
    
#隐含信息分布对应的图片打印出来
    my_con=tf.placeholder(tf.float32, [batch_size,2])
    myz = tf.concat(axis=1, values=[tf.one_hot(y, depth = classes_dim), my_con, z_rand])
    mygen = generator(myz)
    mygenout= tf.squeeze(mygen, -1) 
    
    my_con1 = np.ones([10,2])
    a = np.linspace(0.0001, 0.99999, 10)
    y_input= np.ones([10])
    figure = np.zeros((28 * 10, 28 * 10))
    my_rand = tf.random_normal((10, rand_dim))
    for i in range(10):
        for j in range(10):
            my_con1[j][0]=a[i]
            my_con1[j][1]=a[j]
            y_input[j] = j
        mygenoutv =  sess.run(mygenout,feed_dict={y:y_input,my_con:my_con1})
        for jj in range(10):
            digit = mygenoutv[jj].reshape(28, 28)
            figure[i * 28: (i + 1) * 28,
                   jj * 28: (jj + 1) * 28] = digit
    
    plt.figure(figsize=(10, 10))
    plt.imshow(figure, cmap='Greys_r')
    plt.show() 



