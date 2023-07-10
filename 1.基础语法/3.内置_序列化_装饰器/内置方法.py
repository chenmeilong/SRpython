
# print( all([0,-5,3]) )                  #列表包含0就返回假，都是其他数为真
# print( any([]) )                        #0和空返回假，其他有一个为真即为真
# a= ascii([1,2,"开外挂开外挂"])             #把列表变字符串
# print(type(a),[a])                        #把列表变字符串
# print(bin(255))                            #数字十进制转换2进制
# print(bool([]))                              #可判断数字列表字典真假

# a = bytes("abcde",encoding="utf-8")
# b = bytearray("abcde",encoding="utf-8")          #可以转成ascil输出
# print( b[1] )
# b[1]= 50                                         #可以用ascil修改
# print(b)
# print(a.capitalize(),a)                         #首字母大写

# def sayhi():pass                                #这是一个空函数
# print( callable(sayhi) )                       #判断是否可调用

# print(chr(97))                                    #返回ascil表值
# print(ord('b'))                                    ##返回ascil表值

# code = '''
def fib(max): #10
    n, a, b = 0, 0, 1
    while n < max: #n<10
        #print(b)
        yield b
        a, b = b, a + b
        #a = b     a =1, b=2, a=b , a=2,
        # b = a +b b = 2+2 = 4
        n = n + 1
    return '---done---'

#f= fib(10)
g = fib(6)
while True:
    try:
        x = next(g)
        print('g:', x)
    except StopIteration as e:
        print('Generator return value:', e.value)
        break
#
# '''
# py_obj = compile(code,"err.log","exec")           # err.log可以将运行出现的错误打印出来    编译完用exec执行        compile可以把字符串编译成代码执行
# exec(py_obj)                                      #可以实现动态导入的功能      字符串转代码运行
#
# #exec(code)                                        #这个可以代替前面两句

#print(divmod(7,2))                                   #取商和余数


# name=["zhangsan","lisi","wanger"]                     #加序号 列表转换    常用操作
# name2=list(enumerate(name))
# name3=list(enumerate(name,start=1))
# print(name2)
# print(name3)


# def sayhi(n):                                    #下面匿名函数的原函数
#     print(n)
#     for i in range(n):
#         print(i)
# sayhi(3)
#
#(lambda n:print(n))(5)                        #匿名函数
# calc = lambda n:3 if n<4 else n                #匿名函数  做不了复杂运算
# print(calc(0))

# res = filter(lambda n:n>5,range(10))                 #过滤filter留下n>5的数    后面要循环打印出结果
# res = map(lambda n:n*2,range(10))                    # 可以整体*2   后面循环打印      lambda与map搭配使用 把后面的值给前面的操作，后返回一个列表
# import functools
# res = functools.reduce( lambda x,y:x*y,range(1,10))           #x为运算结果   y是后面循环操作值
# print(res)


#a = frozenset([1,4,333,212,33,33,12,4])                     #添加frozenset()变成了不可变列表



# print(globals())                                            #打印当前程序/文件所有的变量名 和变量值   字典形式
# def test():
#     local_var =333
#     print(locals())                                   #只打印局部变量
#     print(globals())                                  #只打印全局变量
# test()
# print(locals().get('local_var'))
# print(globals().get('local_var'))                     #返回值真假  代表变量定义与否


# print(hex(255))                                              #十进制转十六进制
# print(oct(64))                                                 #十进制转八进制
# print(pow(2,8))                                                 #既2的8次幂
    # reversed 翻转
#print(round(1.321,2))                                            #取小数位数
    #slice  切片

# a = {6:2,8:0,1:4,-5:6,99:11,4:22}
# #print(  sorted(a.items()) )                                  #sorted 可以给key字典排序  排完变列表
# print(  sorted(a.items(),key=lambda x:x[1]) )                 #sorted 可以给value字典排序  排完变列表






# a = [1,2,3,4,5,6,7]
# b = ['a','b','c','d']
#
# for i in zip(a,b):                                         #将两列表组合  zip 拉链组合
#     print(i)

# #import 'decorator'                                         #很常用  寻找字符串模块名
# __import__('decorator')
