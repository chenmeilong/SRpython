class Dog(object):
    name = "我是类变量"                                         #类方法只能访问类变量，不能访问实例变量

    def __init__(self, name):
        self.name = name

    @classmethod                                                #类方法只能访问类变量，不能访问实例变量
    def eat(self):
        print("%s is eating" % self.name)


d = Dog("ChenRonghua")
d.eat()