class Role (object):               #类名role    (object)  可以不写
    n = 123                        #类变量           节省空间
    name="类变量"                  #如果实例变量没有，就会去找类变量 输出
    def __init__(self, name, role, weapon, life_value=100, money=15000):                #传参数   self接收变量名  相当于后面的r1 r1
        #构造函数
        #在实例化时做一些类的初始化的工作
        self.name = name                          #相当于nr1.name=name实例变量(静态属性),作用域就是实例本身
        self.role = role                          #实例变量
        self.weapon = weapon
        self.life_value = life_value
        self.money = money

    def shot(self):                              #  类的方法，功能 （动态属性）
        print("shooting...")

    def got_shot(self):
        print("ah...,I got shot...")

    def buy_gun(self, gun_name):
        print("%s just bought %s" %(self.name,gun_name))

r1=Role('Alex', 'police', 'AK47')                 #生成一个角色  role的实例
r1.got_shot()                                     #把人杀死        调用class

r2 = Role('Jack', 'terrorist','B22')              #把一个类变成一个具体对象的过程叫 实例化(初始化一个类，造了一个对象)

r1.buy_gun("AK47")

print(r1.name)                             #如果实例变量没有，就会去找类变量 输出


r3=Role('chen', 'police', 'AK47')                 #可以操作类的储存  修改参数
r3.name='long'
print(r3.name)
r3.bullet_prove = True                              # 实例化完成后 可以  加类属性  防弹衣       只是在r3才有
print(r3.bullet_prove)

r1.n = "改类变量"                                  #只在r1里面改类变量
print("r1:",r1.weapon,r1.n )
print("r2:",r2.weapon,r2.n )

Role.n = "改类变量"                                  #都改了变量
print("r1:",r1.weapon,r1.n )
print("r2:",r2.weapon,r2.n )


