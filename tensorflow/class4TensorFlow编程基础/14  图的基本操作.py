import numpy as np
import tensorflow as tf 

# 1 创建图的方法
c = tf.constant(0.0)      #默认图

g = tf.Graph()           #创建图
with g.as_default():      #  使用tf.Graph() 函数创建图，并在上面定义op
  c1 = tf.constant(0.0)
  print(c1.graph)       #输出新图  的 新变量图的位置
  print(g)
  print(c.graph)

g2 =  tf.get_default_graph()     #获取默认图
print(g2)

tf.reset_default_graph()      # 重置默认图
g3 =  tf.get_default_graph()
print(g3)                     #输出重置后的默认图

# 2.	获取tensor

print(c1.name)      #输出名字
t = g.get_tensor_by_name(name = "Const:0")    #根据名字获取tensor
print(t)

# 3 获取op
a = tf.constant([[1.0, 2.0]])
b = tf.constant([[1.0], [3.0]])

tensor1 = tf.matmul(a, b, name='exampleop')      #a*b=7    重点  ：这里是张量不是op  op是描述张量中的运算关系的
print(tensor1.name,tensor1) 
test = g3.get_tensor_by_name("exampleop:0")     #获取张量
print(test)

print(tensor1.op.name)      #输出节点操作 ：乘
testop = g3.get_operation_by_name("exampleop")      #获取节点op
print(testop)
print("________________________\n")

with tf.Session() as sess:
    test =  sess.run(test)   #输出运行张量结果
    print(test) 
    test = tf.get_default_graph().get_tensor_by_name("exampleop:0")
    print (test) 

#4 获取所有列表

#返回图中的操作节点列表
tt2 = g3.get_operations()
print(tt2)         #g3图中有常量  有mul乘
#5  获取张量对象
tt3 = g.as_graph_element(c1)
print(tt3)
print("________________________\n")


#练习
with g.as_default():
 c1 = tf.constant(0.0)
 print(c1.graph)
 print(g)
 print(c.graph)    #仍然是全局图
 g3 = tf.get_default_graph()    #g3的默认图变成了，这里新建的局部图
 print(g3)
