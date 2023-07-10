import tensorflow as tf
from numpy.random import RandomState
import pandas as pd
import numpy as np
import time

#定义神经网络的参数
d=3#输入节点个数
l=1#输出节点个数
q=2*d+1#隐层个数,采用经验公式2d+1
train_num=480#训练数据个数
test_num=240#测试数据个数
eta=0.5#学习率
error=0.002#精度

#初始化权值和阈值
w1= tf.Variable(tf.random_normal([d, q], stddev=1, seed=1))#seed设定随机种子，保证每次初始化相同数据
b1=tf.Variable(tf.constant(0.0,shape=[q]))
w2= tf.Variable(tf.random_normal([q, l], stddev=1, seed=1))
b2=tf.Variable(tf.constant(0.0,shape=[l]))

#输入占位
x = tf.placeholder(tf.float32, shape=(None, d))#列数是d，行数不定
y_= tf.placeholder(tf.float32, shape=(None, l))

#构建图：前向传播
a=tf.nn.sigmoid(tf.matmul(x,w1)+b1)#sigmoid激活函数
y=tf.nn.sigmoid(tf.matmul(a,w2)+b2)
mse = tf.reduce_mean(tf.square(y_ -  y))#损失函数采用均方误差
train_step = tf.train.AdamOptimizer(eta).minimize(mse)#Adam算法
#train_step = tf.train.GradientDescentOptimizer(eta).minimize(mse)#梯度下降法

#读取气温数据
dataset = pd.read_csv('tem.csv', delimiter=",")
dataset=np.array(dataset)
m,n=np.shape(dataset)
totalX=np.zeros((m-d,d))
totalY=np.zeros((m-d,l))
for i in range(m-d):#分组：前三个值输入，第四个值输出
    totalX[i][0]=dataset[i][0]
    totalX[i][1]=dataset[i+1][0]
    totalX[i][2]=dataset[i+2][0]
    totalY[i][0]=dataset[i+3][0]
#归一化数据
Normal_totalX=np.zeros((m-d,d))
Normal_totalY=np.zeros((m-d,l))
nummin=np.min(dataset)
nummax=np.max(dataset)
dif=nummax-nummin
for i in range(m-d):
    for j in range(d):
        Normal_totalX[i][j]=(totalX[i][j]-nummin)/dif
    Normal_totalY[i][0]=(totalY[i][0]-nummin)/dif

#截取训练数据
X=Normal_totalX[:train_num-d,:]
Y=Normal_totalY[:train_num-d,:]
testX=Normal_totalX[train_num:,:]
testY=totalY[train_num:,:]
start = time.clock()

#创建会话来执行图
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()#初始化节点
    sess.run(init_op)

    STEPS=0
    while True:
        sess.run(train_step, feed_dict={x: X, y_: Y})
        STEPS+=1
        train_mse= sess.run(mse, feed_dict={x: X, y_: Y})
        if STEPS % 10 == 0:#每训练100次，输出损失函数
            print("第 %d 次训练后,训练集损失函数为：%g" % (STEPS, train_mse))
        if train_mse<error:
            break
    print("总训练次数：",STEPS)
    end = time.clock()
    print("运行耗时(s)：",end-start)
	
	#测试
    Normal_y= sess.run(y, feed_dict={x: testX})#求得测试集下的y计算值
    DeNormal_y=Normal_y*dif+nummin#将y反归一化
    test_mse= sess.run(mse, feed_dict={y: DeNormal_y, y_: testY})#计算均方误差
    print("测试集均方误差为：",test_mse)
	
	#预测
    XX=tf.constant([[18.3,17.4,16.7]])
    XX=(XX-nummin)/dif#归一化
    a=tf.nn.sigmoid(tf.matmul(XX,w1)+b1)
    y=tf.nn.sigmoid(tf.matmul(a,w2)+b2)
    y=y*dif+nummin#反归一化
    print("[18.3,17.4,16.7]输入下,预测气温为：",sess.run(y))