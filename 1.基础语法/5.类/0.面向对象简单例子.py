class Dog:
    def __init__(self,name):                                 #传名字
        self.name = name

    def bulk(self):
        print("%s: wang wang wang!" % self.name)


d1 = Dog("张三")
d2 = Dog("李四")
d3 = Dog("王二")

d1.bulk()
d2.bulk()
d3.bulk()