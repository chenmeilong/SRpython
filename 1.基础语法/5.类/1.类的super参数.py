
# #提出问题
# class Base:
#     def __init__(self):
#         print('Base.__init__')
# class A(Base):
#     def __init__(self):
#         Base.__init__(self)
#         print('A.__init__')
# class B(Base):
#     def __init__(self):
#         Base.__init__(self)
#         print('B.__init__')
# class C(A, B):
#     def __init__(self):
#         A.__init__(self)
#         B.__init__(self)
#         print('C.__init__')
# C()         #继承了两遍   Base  这不是我们想要的


#解决办法 ， 使用super继承
class Base:
    def __init__(self):
        print('Base.__init__')
class A(Base):
    def __init__(self):
        super().__init__()       #在定义类当中可以不写参数，Python会自动根据情况将两个参数传递给super
        print('A.__init__')
class B(Base):
    def __init__(self):
        super().__init__()
        print('B.__init__')
class C(A, B):
    def __init__(self):
        print(super())       #在定义类当中可以不写参数，Python会自动根据情况将两个参数传递给super
        super(C,self).__init__()  #super().__init__()是一样的
        # super(A, self).__init__()  #Python是按照第二个参数来计算MRO，这次的参数是self，也就是C的MRO。在这个顺序中跳过一个参数（A）找后面一个类（B）
        print('C.__init__')
C()      #继承了一遍Base
print(C.mro())            #继承方式：广度优先原则    因为：每定义一个类的时候，Python都会创建一个MRO列表，用来管理类的继承顺序。

print("#"*100)
c_obj = C()
print(super(C, C))
print(super(C, c_obj))

c1 = super(C, C)
c2 = super(C, C)
print(c1 is c2)
print(c1 == c2)
