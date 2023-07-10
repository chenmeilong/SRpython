'''
import json
f = open("test1","r")
data = json.loads(f.read())
f.close()
print(data['age'])
'''
'''
import pickle
def sayhi(name):                                       #不加这个反序列化报错
    print("hello2,",name)                              #反序列化内容可以修改
    
f = open("test1","rb")
data = pickle.loads(f.read())
#data = pickle.load(f)                                 #data = pickle.loads(f.read())与这句给等效    
f.close()
print(data)
print(data["func"]("Alex"))                              #反序列化内容修改Alex
'''
'''
import json
f = open("test1","r")
for line in f:
    print(line)                                            #dump两次也可
    #print(json.loads(line))                              #dump多次，但是只能load一次  不然会出错
'''