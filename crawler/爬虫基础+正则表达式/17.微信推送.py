"""
Author：'Wayne'
Time：'2021-10-18 13:52'
Email：'chenmeilong1998@foxmail.com'
Function：
"""
#coding:utf-8
import requests
import json
headers={
    "User-Agent":"Mozilla/5.0 (Linux; Android 9; SM-A102U) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.93 Mobile Safari/537.36",
    'Content-Type': 'application/json'
}
url='http://pushplus.hxtrip.com/send'
data={'token':'a5985c0164ed48029a7cb2ae03469367','title':'出校申请','content':'成功','template':'html'}
res=requests.post(headers=headers,url=url,data=json.dumps(data),timeout=10)
print(res.status_code)
print(res.text)
