str="我爱学python"
print(str)
print(str.encode())    #linux会报错

print(str.encode('utf-8'))                   #待测试

print(str.encode(encoding="utf-8"))             #编码
code=b'\xe6\x88\x91\xe7\x88\xb1\xe5\xad\xa6python'
print(code.decode(encoding="utf-8"))                #解码