class Animal:
    def __init__(self, name):                     # Constructor of the class
        self.name = name

    def talk(self):                               # Abstract method, defined by convention only
        pass                      #raise NotImplementedError("Subclass must implement abstract method")   异常处理会讲

    @staticmethod                             #装饰器
    def animal_talk(obj):                      #多态
        obj.talk()

class Cat(Animal):
    def talk(self):
        print('Meow!')


class Dog(Animal):
    def talk(self):
        print('Woof! Woof!')


d = Dog("wang")
# d.talk()

c = Cat("miao")
# c.talk()
# animal_talk(c)
# animal_talk(d)

Animal.animal_talk(c)                       #同种接口多种形态
Animal.animal_talk(d)

