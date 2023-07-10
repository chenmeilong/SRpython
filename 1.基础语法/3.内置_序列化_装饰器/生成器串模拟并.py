import time                                                #典型的生产者，消费者模型
def consumer(name):
    print("%s 准备吃包子啦!" %name)
    while True:
       baozi = yield                                        #保存当前状态并返回，类似于中断

       print("包子[%s]来了,被[%s]吃了!" %(baozi,name))

c = consumer("ChenRonghua")
c.__next__()

# b1= "韭菜馅"
# c.send(b1)
# c.__next__()

def producer(name):
    c = consumer('A')                                          #这里仅是调用函数头   把函数变成生成器
    c2 = consumer('B')                                         #这里仅是调用函数头
    c.__next__()                                               #初始化
    c2.__next__()                                              #初始化
    print("老子开始准备做包子啦!")
    for i in range(10):
        time.sleep(1)
        print("做了1个包子,分两半!")
        c.send(i)                                       #send（）可以给yield传值调用     __next__()只能调用不能传值
        c2.send(i)

producer("alex")