
import tensorflow as tf
import numpy as np

# 网络结构：2维输入 --> 2维隐藏层 --> 1维输出

learning_rate = 1e-4
n_input  = 2
n_label  = 1
n_hidden = 2             #相当于隐藏层的并联细胞个数，这个数越大，训练越准


x = tf.placeholder(tf.float32, [None,n_input])
y = tf.placeholder(tf.float32, [None, n_label])
#定义学习参数  字典法
weights = {
    'h1': tf.Variable(tf.truncated_normal([n_input, n_hidden], stddev=0.1)),   #隐藏层
    'h2': tf.Variable(tf.random_normal([n_hidden, n_label], stddev=0.1))        #输出层
	} 
biases = {
    'h1': tf.Variable(tf.zeros([n_hidden])),
    'h2': tf.Variable(tf.zeros([n_label]))
    }    

#定义网络模型
layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['h1']), biases['h1']))    #relu函数激活
y_pred = tf.nn.tanh(tf.add(tf.matmul(layer_1, weights['h2']),biases['h2']))
#y_pred = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['h2']),biases['h2']))#陷入局部最优解
#y_pred = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['h2']),biases['h2']))

#Leaky relus  40000次 ok
#layer2 =tf.add(tf.matmul(layer_1, weights['h2']),biases['h2'])
#y_pred = tf.maximum(layer2,0.01*layer2)
  
loss=tf.reduce_mean((y_pred-y)**2)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

#生成数据
X=[[0,0],[0,1],[1,0],[1,1]]
Y=[[0],[1],[1],[0]]
X=np.array(X).astype('float32')
Y=np.array(Y).astype('int16')

#加载
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

#训练
for i in range(10000):
    sess.run(train_step,feed_dict={x:X,y:Y} )

     
#计算预测值
print(sess.run(y_pred,feed_dict={x:X}))
#输出：已训练100000次

       
#查看隐藏层的输出
print(sess.run(layer_1,feed_dict={x:X}))