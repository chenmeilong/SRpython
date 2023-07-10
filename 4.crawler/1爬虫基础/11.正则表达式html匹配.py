import re
import urllib.request

response=urllib.request.urlopen("https://tieba.baidu.com/p/4863860271")
html=response.read()
html=html.decode("utf-8")      #解码


p = r'<img class="BDE_Image" src="[^"]+\.jpg"'         # 正常匹配
#p = r'<img class="BDE_Image" src="([^"]+\.jpg)"'          #用小括号分组 只输出括号内的内容

imglist = re.findall(p, html)

for each in imglist:
    print(each)

