#队列数据只有一份  取走就没了
import queue

q  = queue.Queue()     #先入先出
q.put(1)
q.put(2)
q.put(3)
print(q.get())
print(q.get())
print(q.get())

# q = queue.PriorityQueue()   #生成对象  可以设置优先级队列
# q.put((-1,"chenronghua"))   #放数据
# q.put((3,"hanyang"))
# q.put((10,"alex"))
# q.put((6,"wangsen"))
# print(q.get())        #取第一个存的数据
# print(q.get())
# print(q.get())
# print(q.get())

# q  = queue.LifoQueue()     #后入先出
# q.put(1)
# q.put(2)
# q.put(3)
# print(q.get())
# print(q.get())
# print(q.get())

