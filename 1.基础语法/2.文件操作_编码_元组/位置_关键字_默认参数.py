
def test(x,y,z):
    print(x)
    print(y)
    print(z)

# test(y=2,x=1) #与形参顺序无关                              #形式参数，实际参数 一一对应，    形式参数不占用内存只有在调用时，占用内存
# test(1,2)  #与形参一一对应
#test(x=2,3)
test(3,z=2,y=6)

'''
                                                         #默认参数        可修改
def test(x,y=2):                                #默认参数特点：调用函数的时候，默认参数非必须传递
    print(x)                                    #用途      装软件时 ，一键安装和自定义安装
    print(y)
test(1)
test(1,3)
test(1,y=4)
'''





