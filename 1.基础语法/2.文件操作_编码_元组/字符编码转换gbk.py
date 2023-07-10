#-*-coding:gbk-*-                               #文件编译需要生成什么字符格式就写什么字符声明，默认为编译成utf8 不需要声明
                                                #现在看见的数据都是unicode格式储存
import sys
print(sys.getdefaultencoding())
__author__ = "Alex Li"

s = "你哈"                                          #其实储存的还是unicode
print(s.encode("gbk"))                               #gbk编码      即将unicode转换成gbk
print(s.encode("utf-8"))                              #utf8编码
print(s.encode("utf-8").decode("utf-8").encode("gb2312").decode("gb2312"))
