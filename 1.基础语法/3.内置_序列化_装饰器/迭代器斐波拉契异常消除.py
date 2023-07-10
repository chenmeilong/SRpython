# b=(a*3 for a in range(5))                     #迭代器  只在调用时生成
# for i in b:
#     print(i)


def fib(max): #10
    n, a, b = 0, 0, 1
    while n < max: #n<10
        #print(b)
        yield b                    #修改这变成生成器              这里类似于中断  跳到下面19行
        a, b = b, a + b            #相当于  t=（b,a+b）     a=t[0]   b=t(1)
        n = n + 1
    return '--done--'                  #异常打印操作
g = fib(6)
# f=fib(4)
# print(f.__next__())
# print("----------")                     #可以让函数停住，暂停，做其他的事
# print(f.__next__())
# print(f.__next__())
# print(f.__next__())
# print("----------")                     #可以让函数停住，暂停，做其他的事
# print(f.__next__())



while True:                               #消除异常操作
    try:
        x = next(g)                                     #与__next__()效果一样
        print('g:', x)
    except StopIteration as e:                       #指定异常错误，e为名字
        print('Generator return value:', e.value)               #出现异常就打印这个
        break

