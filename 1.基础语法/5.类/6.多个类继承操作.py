class People(object):        #新式类   object可以省略
    def __init__(self,name,age):
        self.name = name
        self.age = age
    def eat(self):
        print("%s is eating..." % self.name)
    def talk(self):
        print("%s is talking..." % self.name)
    def sleep(self):
        print("%s is sleeping..." % self.name)

class Relation(object):                                                      #没有构造变量所以后面不能先调
    def make_friends(self,obj):                                               #w1
        print("%s is making friends with %s" % (self.name,obj.name))

class Man(People,Relation):                                              #子类继承父类   多继承  调用时 重复的可以不写
    def __init__(self,name,age,money):                                  #构造参数 初始化函数   覆盖父类，
        People.__init__(self,name,age)                                     #继承父类
        #super(Man,self).__init__(name,age)                               #新式类写法    推荐使用
        self.money  = money                                                 #增加新的内容
        print("%s 一出生就有%s money" %(self.name,self.money))
    def piao(self):
        print("%s is piaoing ..... 20s....done." % self.name)
    def sleep(self):
        People.sleep(self)                                              #调用父类方法
        print("man is sleeping ")
class Woman(People,Relation):                                            #继承父类     两个父类 多继承
    def get_birth(self):
        print("%s is born a baby...." % self.name)

m1 = Man("NiuHanYang",22,20)
w1 = Woman("ChenRonghua", 26)

m1.make_friends(w1)

