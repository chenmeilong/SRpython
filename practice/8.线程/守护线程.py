# 正常情况会等子现场执行完才退出
#####主进程执行完就退出，不必等子进程执行完成
import threading
import time

def run(n):
    print("task ",n )
    time.sleep(2)
    print("task done",n,threading.current_thread())

start_time = time.time()
t_objs = [] #存线程实例
for i in range(50):
    t = threading.Thread(target=run,args=("t-%s" %i ,))
    t.setDaemon(True) #把当前线程设置为守护线程
    t.start()
    t_objs.append(t) #为了不阻塞后面线程的启动，不在这里join，先放到一个列表里

# time.sleep(2)
print("----------all threads has finished...",threading.current_thread(),threading.active_count())
print("cost:",time.time() - start_time)
