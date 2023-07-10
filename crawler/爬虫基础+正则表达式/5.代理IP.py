#使用代理
import urllib.request
import random

url ="https://www.baidu.com/"
iplist = ['118.190.145.138:9001', '117.191.11.77:8080', '101.4.136.34:8080']
proxy_support = urllib.request.ProxyHandler({'http':random.choice(iplist)})

opener = urllib.request.build_opener(proxy_support)  #创建一个包   含IP的opener
opener.addheader = [('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.142 Safari/537.36')]  #隐藏

urllib.request.install_opener(opener)     #将定制好的opener 安装到系统中

response = urllib.request.urlopen(url)
html = response.read().decode('utf-8')

print(html)
