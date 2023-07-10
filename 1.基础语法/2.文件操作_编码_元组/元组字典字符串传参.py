#*args:接受N个位置参数，转换成元组形式
def test(*args):                #args是一个变量名/形参名  可以修改  但是常用*args    前面加*可以接受 多分实参
    print(args)                 #args接受位置参数转换成元组

test(1,2,3,4,5,5)
test(*[1,2,4,5,5])             #  相当于   args=tuple([1,2,3,4,5])
test()

def test1(x,*args):                #也可以混合用               常用
    print(x)
    print(args)
test1(1,2,3,4,5,6,7)

                               # **kwargs：接受N个  关键字  参数，转换成字典的方式
def test2(**kwargs):
    print(kwargs)
    print(kwargs['name'])
    print(kwargs['age'])
    print(kwargs['sex'])
test2(name='alex',age=8,sex='F')
test2(**{'name':'alex','age':8,'sex':'F'})
def test3(name,**kwargs):
    print(name)
    print(kwargs)
test3('alex',age=18,sex='m')                      #**kwargs 传 关键字

print("-----------------------")

def test4(name,age=18,**kwargs):
    print(name)
    print(age)
    print(kwargs)
    #logger("TEST4")
test4('alex',age=15,hobby='tesla')
test4('alex',age=34,sex='m',hobby='tesla')


                                                    #元组字典混合传参   #arg、*args、**kwargs三个参数的位置必须是一定的。必须是(arg,*args,**kwargs)这个顺序，否则程序会报错。
def test4(name,age=18,*args,**kwargs):
    print(name)
    print(age)
    print(args)
    print(kwargs)
test4('alex',age=37,sex='m',hobby='tesla')

def logger(source):                                 #字符串传参
    print("from %s" %  source)
logger("TEST4")