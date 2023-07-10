import urllib.request
import urllib.error

# HTTPError 是  URLError 的子类

req = urllib.request.Request("http://www.cug123.edu.cn/")    # URLError 的域名
try:
    urllib.request.urlopen(req)
except urllib.error.URLError as e:
    print(e.reason)  # 异常属性



req = urllib.request.Request("http://www.fishc.com/ooxx.html")   #HTTPError 的域名
try:
    urllib.request.urlopen(req)
except urllib.error.HTTPError as e:
    print(e.code)
    print(e.read())
    print(e.reason)


