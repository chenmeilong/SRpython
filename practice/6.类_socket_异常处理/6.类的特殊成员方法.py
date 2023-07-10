#__doc__　　表示类的描述信息
class Foo:
    """ 描述类信息，这个类的作用 """
    def func(self):
        pass
print (Foo.__doc__)


#　　__module__ 表示当前操作的对象在那个模块
#　　__class__     表示当前操作的对象的类是什么
# from moduleclass import C
# obj = C()
# print (obj.__module__)           # 输出moduleclass ，即：输出模块
# print (obj.__class__ )           # 输出 <class 'moduleclass.C'>，即：输出类


#  __call__ 对象后面加括号，触发执行。
class Foo:                             #构造方法的执行是由创建对象触发的，即：对象 = 类名() ；而对于
                                       #__call__方法的执行是由对象后加括号触发的，即：对象() 或者 类()()
    def __init__(self):
        pass
    def __call__(self, *args, **kwargs):                #列表，字典
        print('running call',args,kwargs)
obj = Foo()                      # 执行 __init__
obj()                            # 执行 __call__ 对象()
obj(1,2,3,name='abc')            #这样调用


# #__dict__ 查看类或对象中的所有成员 　　
# class Province:
#     country = 'China'
#     def __init__(self, name, count):
#         self.name = name
#         self.count = count
#     def func(self, *args, **kwargs):
#         print('func')
# # 获取类的成员，即：静态字段、方法、
# print (Province.__dict__)
# # 输出：{'country': 'China', '__module__': '__main__', 'func': <function func at 0x10be30f50>, '__init__': <function __init__ at 0x10be30ed8>, '__doc__': None}
# obj1 = Province('HeBei', 10000)
# print (obj1.__dict__)
# # 获取 对象obj1 的成员
# # 输出：{'count': 10000, 'name': 'HeBei'}
# obj2 = Province('HeNan', 3888)
# print (obj2.__dict__)
# # 获取 对象obj1 的成员
# # 输出：{'count': 3888, 'name': 'HeNan'}


# #__str__ 如果一个类中定义了__str__方法，那么在打印 对象 时，默认输出该方法的返回值。
# class Foo:
#     def __str__(self):
#         return 'chen meilong'
# obj = Foo()
# print(obj)
# # 输出：alex li


# # __getitem__、__setitem__、__delitem__        很少用
# # 用于索引操作，如字典。以上分别表示获取、设置、删除数据
# class Foo(object):
#     def __getitem__(self, key):
#         print('__getitem__', key)
#
#     def __setitem__(self, key, value):
#         print('__setitem__', key, value)
#
#     def __delitem__(self, key):
#         print('__delitem__', key)
# obj = Foo()
# result = obj['k1']        # 自动触发执行 __getitem__
# obj['k2'] = 'alex'        # 自动触发执行 __setitem__
# del obj['k2']


