'''
import urllib.request as ur
from lxml import etree
import zlib,chardet
url = "https://www.kuaidaili.com/free/"
header = {
    #浏览器信息
    'User-Agent': 'Mozilla/4.0 (compatible; MSIE 5.5; Windows NT)'
}
request = ur.Request(url,headers=header)
html_str = ur.urlopen(request).read()
#解压
htmlencode = zlib.decompress(html_str, 16+zlib.MAX_WBITS)
html_str_code = htmlencode.decode("utf-8")
print(html_str_code)
html = etree.HTML(html_str_code)
result_ip = html.xpath('//tr/td[@data-title="IP"]/text()')
result_port = html.xpath('//tr/td[@data-title="PORT"]/text()')
print(result_ip)
print(result_port)
arr = []
for i in range(len(result_ip)):
    ip_port = result_ip[i]+":"+result_port[i]
    arr.append(ip_port)
print(arr)
import random
proxy_support = ur.ProxyHandler({"http":random.choice(arr)})
opener = ur.build_opener(proxy_support
ur.install_opener(opener)
'''
import telnetlib
try:
    telnetlib.Telnet('139.196.90.80', port='80', timeout=5)
except:
    print('connect failed')
else:
    print('success')
