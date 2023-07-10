import  time                                 #调用time模块  后面有打印当前时间操作
def logger():
    time_format = '%Y-%m-%d %X'                                #年月日      时间  格式
    time_current = time.strftime(time_format)                   #调用当前时间操作
    print(time_current)
    with open('a.txt','a+') as f:
        f.write('%s end action\n' %time_current)

def test1():
    print('in the test1')

    logger()
def test2():
    print('in the test2')

    logger()
def test3():
    print('in the test3')
    logger()

test1()
test2()
test3()
