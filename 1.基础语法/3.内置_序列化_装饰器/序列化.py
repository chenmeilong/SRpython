'''
import json                                         #只能处理简单的，json是所有语言通用的
info = {
    'name':'alex',
    'age':22,
}
f = open("test1","w")
#print(json.dumps(info))                                    #json可以将字典变字符串
f.write( json.dumps( info) )
f.close()
'''
'''
import pickle                                               #有自己的储存字符码，所以teat1 会乱码
def sayhi(name):
    print("hello,",name)
info = {
    'name':'alex',
    'age':22,
    'func':sayhi
}
f = open("test1","wb")                                    #以二进制形式写入
f.write( pickle.dumps( info) )
#pickle.dump(info,f)                            #f.write( pickle.dumps( info) )  与这句话等效
f.close()
'''
'''
import json
def sayhi(name):
    print("hello,",name)
info = {
    'name':'alex',
    'age':22,
}
f = open("test1","w")
f.write( json.dumps( info) )
info['age'] = 21
f.write( json.dumps( info) )
f.close()
'''