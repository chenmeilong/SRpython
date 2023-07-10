class People:        #新式类
    def __init__(self,name,age):
        self.name = name
        self.age = age
    def eat(self):
        print("%s is eating..." % self.name)
    def talk(self):
        print("%s is talking..." % self.name)
    def sleep(self):
        print("%s is sleeping..." % self.name)

class Man(People):                                              #子类继承父类
    def piao(self):
        print("%s is piaoing ..... 20s....done." % self.name)
    def sleep(self):
        People.sleep(self)                                  #调用父类方法
        print("man is sleeping ")
class Woman(People):                                  #继承父类
    def get_birth(self):
        print("%s is born a baby...." % self.name)


m1=Man("chen",18)
m1.eat()
m1.piao()
m1.sleep()

w1= Woman("ChenRonghua",26)
w1.get_birth()
w1.eat()

