import tensorflow as tf
tf.reset_default_graph() 

    
#var1 = tf.get_variable("firstvar",shape=[2],dtype=tf.float32)   
#var2 = tf.get_variable("firstvar",shape=[2],dtype=tf.float32)    
    
with tf.variable_scope("test1", ):
    var1 = tf.get_variable("firstvar",shape=[2],dtype=tf.float32)
    
    with tf.variable_scope("test2"):
        var2 = tf.get_variable("firstvar",shape=[2],dtype=tf.float32)
        
print ("var1:",var1.name)
print ("var2:",var2.name)


with tf.variable_scope("test1",reuse=True ):    #    ture  变量共享
    var3= tf.get_variable("firstvar",shape=[2],dtype=tf.float32)
    with tf.variable_scope("test2"):
        var4 = tf.get_variable("firstvar",shape=[2],dtype=tf.float32)

print ("var3:",var3.name)
print ("var4:",var4.name)

