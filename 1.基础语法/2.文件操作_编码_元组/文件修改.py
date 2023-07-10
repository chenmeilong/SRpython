'''                              #标准格式
f = open("yesterday2","r",encoding="utf-8")
f_new = open("yesterday2.bak","w",encoding="utf-8")
for line in f:
    if"肆意的快乐等我享受"in line:
        line = line.replace("肆意的快乐等我享受","肆意的快乐等Chen享受")
    f_new.write(line)
f.close()
f_new.close()
'''
'''
                          #简单练习
a=open("yesterday2",'r',encoding="utf-8")
b=open("yesterday3",'w',encoding='utf-8')
for c in a:
    if'舌尖'in c:
        c=c.replace('舌尖','嘴尖')
    b.write(c)
a.close()
b.close()
'''
'''                 #格式        
find_str = sys.argv[1]
replace_str = sys.argv[2]
for line in f:
    if find_str in line:
        line = line.replace(find_str,replace_str)
    f_new.write(line)
f.close()
f_new.close()
'''

'''
import sys                         #需要模块调用，传入参数才能用
f = open("yesterday2","r",encoding="utf-8")
f_new = open("yesterday2.bak","w",encoding="utf-8")

find_str = sys.argv[1]
replace_str = sys.argv[2]
for line in f:
    if find_str in line:
        line = line.replace(find_str,replace_str)
    f_new.write(line)
f.close()
f_new.close()
'''