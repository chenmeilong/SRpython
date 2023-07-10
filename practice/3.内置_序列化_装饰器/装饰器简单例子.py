import time
def timmer(func):                                      #装饰简单例子
    def warpper(*args,**kwargs):
        start_time=time.time()
        func()
        stop_time=time.time()
        print('the func run time is %s' %(stop_time-start_time))
    return warpper
@timmer                                                 #特点：不修改源代码，不修改调用方式
def test1():
    time.sleep(1)
    print('in the test1')

test1()