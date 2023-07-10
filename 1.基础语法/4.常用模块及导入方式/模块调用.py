# import module_chen                                  #方法一
# print(module_chen.name)                            #模块调用
# module_chen.say_hello()


# from module_chen import *                            #x导入所有内容             方法二
# print(name)                                          #模块调用
# say_hello()                                          #相当于将所有东西导入过来直接用


# from module_chen import say_hello                          #只调用相应函数，可以提高效率
# say_hello()


# from module import module_long                     #同级文件夹内模块调用
# print(module_long.name)                            #模块调用
# module_long.say_hello()


# import sys,os                                               #跨文件调用模块
# x=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))                  #返回上级 目录
# sys.path.append(x)
# from practice4 import 嵌套函数                     #同级文件夹内模块调用
# 嵌套函数.foo()



# import module                                         #导入文件夹里的__init__文件
# module.module_long.logger()



