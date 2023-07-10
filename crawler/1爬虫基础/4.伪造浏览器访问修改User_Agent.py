#隐藏  伪造浏览器访问  修改User-Agent的两种方法   1.实例化Request对象时将headers参数传入 2.通过add_header 方法往Request对象添加headers

import urllib.request
import urllib.parse
import json
import time

while True:
    content = input('请输入待翻译的内容（输入"q!"退出程序）：')
    if content == 'q!':
        break
    
    url = "http://fanyi.youdao.com/translate?smartresult=dict&smartresult=rule&smartresult=ugc&sessionFrom=http://www.youdao.com/"
    #head = {}    #法1  必须是一个字典
    #head['User-Agent'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.142 Safari/537.36'    #伪造浏览器信息   #法1

    data = {}
    data['type'] = 'AUTO'
    data['i'] = content
    data['doctype'] = 'json'
    data['xmlVersion'] = '1.6'
    data['keyfrom'] = 'fanyi.web'
    data['ue'] = 'UTF-8'
    data['typoResult'] = 'true'
    data = urllib.parse.urlencode(data).encode('utf-8')

    #req = urllib.request.Request(url, data,head)     #法1
    req = urllib.request.Request(url, data)           #法二
    req.add_header('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.142 Safari/537.36')   #法2

    response = urllib.request.urlopen(req)
    html = response.read().decode('utf-8')

    target = json.loads(html)
    target = target['translateResult'][0][0]['tgt']

    print(target)

    time.sleep(5)    #  访问太快会被拉黑  所以延时5s
