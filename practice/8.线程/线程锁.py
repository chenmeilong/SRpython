import threading
import time

def run(n):
    lock.acquire()    #获取线程锁
    global  num
    num +=1
    lock.release()    #释放锁  要立即释放
    time.sleep(1)

lock = threading.Lock()
num = 0
t_objs = [] #存线程实例
for i in range(50):
    t = threading.Thread(target=run,args=("t-%s" %i ,))
    t.start()
    t_objs.append(t) #为了不阻塞后面线程的启动，不在这里join，先放到一个列表里

for t in t_objs: #循环线程实例列表，等待所有线程执行完毕
    t.join()

print("----------all threads has finished...",threading.current_thread(),threading.active_count())

print("num:",num)