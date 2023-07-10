import time
def timer(func):                       #timer(test1)  func=test1         嵌套高阶函数修饰
    def deco(*args,**kwargs):               #加了两个非固定参数
        start_time=time.time()
        func(*args,**kwargs)   #run test1()
        stop_time = time.time()
        print("the func run time  is %s" %(stop_time-start_time))
    return deco
@timer               #相当于这个操作test1=timer(test1)   详见高阶函数
def test1():
    time.sleep(1)
    print('in the test1')

@timer # test2 = timer(test2)  = deco  test2(name) =deco(name)
def test2(name,age):
    print("test2:",name,age)

test1()
test2("alex",22)