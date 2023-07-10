# 静态方法
#     只是名义上归类管理， 实际上在静态方法里访问不了类或实例中的任何属性
# 类方法
#     只能访问类变量，不能访问实例变量
#
# 属性方法
#     把一个方法变成一个静态属性
#    flight.status
#     @status.setter
#     flight.status = 3
#     @status.delter
#
# 反射
#     hasattr(obj,name_str) , 判断一个对象obj里是否有对应的name_str字符串的方法
#     getattr(obj,name_str), 根据字符串去获取obj对象里的对应的方法的内存地址
#     setattr(obj,'y',z), is equivalent to ``x.y = v''
#     delattr
#
#
# 异常
#     try :
#         code
#     except (Error1,Erro2) as e:
#         print e
#
#     except Exception :抓住所有错误，不建议用
#
#
#
# Socket网络编程