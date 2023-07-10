class chenError(Exception):               #自定义异常
    def __init__(self, msg):
        self.message = msg
    # def __str__(self):
    #     return 'sdfsf'                            #异常要打印的数据e ，上面Exception传入自动写了，这里可以不用写
try:
    raise chenError('数据库连不上')                 #  数据连不上  传到mag里面   raise触发自己写的异常
except chenError as e:                             #
    print(e)