import urllib.request
response=urllib.request.urlopen("https://www.baidu.com/")
# rep=urllib.request.Request("https://www.baidu.com/")    #法二
# response=urllib.request.urlopen(rep)     #可以是对象也可以是字符串


html=response.read()
print (html)

html=html.decode("utf-8")      #解码
print (html)
