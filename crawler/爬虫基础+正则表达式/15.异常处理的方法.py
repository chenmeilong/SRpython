#先看有没有错误（包括URLError 和 HTTPError ），只要有 其中一个，就会打印 reason， 然后继续判断是否有 code ，如果有 code，就是 HTTPError ，然后也把 code 打印出来。
import urllib.request
import urllib.error
req = urllib.request.Request("http://www.fishc.com/ooxx.html")   ## URLError 的域名  或者    #HTTPError 的域名
try:
	urllib.request.urlopen(req)
except urllib.error.URLError as e:
	if hasattr(e, 'reason'):
		print('Reason: ', e.reason)
	if hasattr(e, 'code'):
		print('Error code: ', e.code)     #HTTPError  会输出http 状态码
