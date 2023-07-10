class Role:
    def __init__(self, name, role, weapon, life_value=100, money=15000):
        #构造函数
        #在实例化时做一些类的初始化的工作
        self.name = name #r1.name=name实例变量(静态属性),作用域就是实例本身
        self.role = role
        self.weapon = weapon
        self.__life_value = life_value
        self.money = money
    def __del__(self):                                                #程序退出就会执行  删除所有实例
        print("%s 彻底死了。。。。" %self.name)

    def show_status(self):
        print("name:%s weapon:%s life_val:%s" %(self.name,
                                                 self.weapon,
                                                self.__life_value))
    def __shot(self):                       # 类的方法，功能 （动态属性）
        print("shooting...")

    def got_shot(self):
        self.__life_value -=50                      #  __加两个下划线 私有属性，   不能修改
        print("%s:ah...,I got shot..."% self.name)

    def buy_gun(self, gun_name):
        print("%s just bought %s" % (self.name,gun_name) )



r1 = Role('Chenronghua', 'police',  'AK47')
r2 = Role('jack', 'terrorist', 'B22')

#print(r1.__init__value)                             #私有属性外部不能修改
r1.show_status()                                     #私有属性在内部可以访问，修改
#r1.__shot()                                          #私有方法 外部不能访问


del r2                                       #   删除 实例r2  执行了 def __del__(self):
