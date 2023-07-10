import sys
print(sys.getdefaultencoding())
__author__ = "Alex Li"

s = "你哈"
s_gbk = s.encode("gbk")

print(s_gbk)                                                      #gbk编码
print(s.encode())                                                 #utf8编码

gbk_to_utf8 = s_gbk.decode("gbk").encode("utf-8")                 #utf8转gbk再转utf8
print("utf8",gbk_to_utf8)