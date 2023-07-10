def add(a,b,f):
    return f(a)+f(b)                         #变量可以指向函数，函数的参数能接收变量，那么一个函数就可以接收另一个函数作为参数，这种函数就称之为高阶函数。
                                             #返回值包含函数名  也叫高阶函数
res = add(3,-6,abs)
print(res)


import time
def bar():
    time.sleep(3)
    print('in the bar')

def test1(func):
    start_time=time.time()
    func()                                              # 运行bar的时间
    stop_time=time.time()
    print("the func run time is %s" %(stop_time-start_time))

test1(bar)                                              #调用方式改变
bar()


# import time
# def bar():
#     time.sleep(3)
#     print('in the bar')
# def test2(func):
#     print(func)                            #输出的是bar地址
#     return func
#
# # print(test2(bar))
# bar=test2(bar)                               #python解释器符号可以解释   后面的方法可以用 @修饰函数名  操作
# bar()                                       # 函数调用方式没改变
