#NumPy 包含大量的各种数学运算的函数，包括三角函数，算术运算的函数，复数处理函数等。
import numpy as np
#三角函数      NumPy 提供了标准的三角函数：sin()、cos()、tan()。
# arcsin，arccos，和 arctan 函数返回给定角度的 sin，cos 和 tan 的反三角函数。
# 这些函数的结果可以通过 numpy.degrees() 函数将弧度转换为角度。
a = np.array([0,30,45,60,90])
print ('含有正弦值的数组：')
sin = np.sin(a*np.pi/180)
print (sin)
print ('\n')
print ('计算角度的反正弦，返回值以弧度为单位：')
inv = np.arcsin(sin)
print (inv)
print ('\n')
print ('通过转化为角度制来检查结果：')
print (np.degrees(inv))
print ('\n')
print ('arccos 和 arctan 函数行为类似：')
cos = np.cos(a*np.pi/180)
print (cos)
print ('\n')
print ('反余弦：')
inv = np.arccos(cos)
print (inv)
print ('\n')
print ('角度制单位：')
print (np.degrees(inv))
print ('\n')
print ('tan 函数：')
tan = np.tan(a*np.pi/180)
print (tan)
print ('\n')
print ('反正切：')
inv = np.arctan(tan)
print (inv)
print ('\n')
print ('角度制单位：')
print (np.degrees(inv))
print("---------------")


#numpy.around() 函数返回指定数字的四舍五入值。
a = np.array([1.0,5.55,  123,  0.567,  25.532])
print  ('原数组：')
print (a)
print ('\n')
print ('舍入后：')
print (np.around(a))
print (np.around(a, decimals =  1))            #decimals舍入的小数位数。 默认值为0。 如果为负，整数将四舍五入到小数点左侧的位置
print (np.around(a, decimals =  -1))
print("---------------")

#numpy.floor() 返回数字的下舍整数。
a = np.array([-1.7,  1.5,  -0.2,  0.6,  10])
print ('提供的数组：')
print (a)
print ('\n')
print ('修改后的数组：')
print (np.floor(a))
print("---------------")

#numpy.ceil() 返回数字的上入整数。
a = np.array([-1.7,  1.5,  -0.2,  0.6,  10])
print  ('提供的数组：')
print (a)
print ('\n')
print ('修改后的数组：')
print (np.ceil(a))

