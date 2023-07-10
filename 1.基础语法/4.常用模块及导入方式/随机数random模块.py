import random
print (random.random())                                                     #随机数  [0,1]  小数
print (random.randint(1,2))                                                 #[1,2]  整数
print (random.randrange(1,10))                                               #[1,10)整数


import random                                                               #生成随机验证码
checkcode = ''
for i in range(4):
    current = random.randrange(0,4)
    if current != i:
        temp = chr(random.randint(65,90))
        print(temp,"temp")
    else:
        temp = random.randint(0,9)
    checkcode += str(temp)
print (checkcode)