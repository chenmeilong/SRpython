def bulk(self):
    print("%s is yelling...." %self.name)

class Dog(object):
    def __init__(self,name):
        self.name = name

    def eat(self,food):
        print("%s is eating..."%self.name,food)


d = Dog("wangxudongh")                                        #对象
choice = input(">>:").strip()



print(hasattr(d,choice))                                   #判断是否字符串存在类里面    d是对象
if hasattr(d,choice):
    func=getattr(d,choice)                                 #调用eat   getattr(d,choice)实质是字符串地址 getattr(d,choice)（）这样调用出来
    func('chenqiang')                                      #方法
else:
    ## setattr(d,choice,bulk)                                #通过字符串设置属性 相当于   x.y=z   x是对象  y是字符串  z是对应的值
    ## d.talk(d)                                             #动态方法
    setattr(d, choice,11)                                   #后面无论输入什么都输出11
    print(getattr(d, choice))                                 #静态属性



#
# if hasattr(d,choice):
#     attr=getattr(d,choice)                                  #改值
#     setattr(d, choice,"hemeng")                             #name改为hemeng
# print(d.name)                                               #改值



# if hasattr(d,choice):
#     delattr(d,choice)                                    #删除
# print(d.name)

