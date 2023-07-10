class Dog(object):

    def __init__(self, name):
        self.name = name

    @staticmethod                   # 把eat方法变为静态方法   实际上跟类没什么关系了
    def eat(self):
        print("%s is eating" % self.name)


d = Dog("ChenRonghua")
# d.eat()                                     #因为是静态方法 调用不了，
d.eat(d)                                      #这样能调用静态方法


# class Dog(object):
#    def __init__(self,name):
#       self.name = name
#
#    @staticmethod
#    def eat():
#        print(" is eating")
#
#
#
# d = Dog("ChenRonghua")
# d.eat()                                        #或者不加参数调用