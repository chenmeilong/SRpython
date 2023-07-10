# -*- coding: utf-8 -*-
import urllib.request
import json
#定义要爬取的微博大V的微博ID
id='6012806447'

#设置代理IP
proxy_addr="122.241.72.191:808"

#定义页面打开函数
def use_proxy(url,proxy_addr):
    req = urllib.request.Request(url)
    req.add_header("User-Agent","Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/49.0.2623.221 Safari/537.36 SE 2.X MetaSr 1.0")
    proxy = urllib.request.ProxyHandler({'http': proxy_addr})
    opener = urllib.request.build_opener(proxy, urllib.request.HTTPHandler)
    urllib.request.install_opener(opener)
    data = urllib.request.urlopen(req).read().decode('utf-8', 'ignore')
    return data

#获取微博大V账号的用户基本信息，如：微博昵称、微博地址、微博头像、关注人数、粉丝数、性别、等级等
def get_userInfo(id):
    url = 'https://m.weibo.cn/api/container/getIndex?type=uid&value=' + id
    data = use_proxy(url, proxy_addr)
    content = json.loads(data).get('data')
    profile_image_url = content.get('userInfo').get('profile_image_url')    #主页地址
    description = content.get('userInfo').get('description')                 #微博说明
    profile_url = content.get('userInfo').get('profile_url')                 #头像地址
    verified = content.get('userInfo').get('verified')                       #认证状态
    guanzhu = content.get('userInfo').get('follow_count')                   #关注
    name = content.get('userInfo').get('screen_name')                       #名字
    fensi = content.get('userInfo').get('followers_count')                  #粉丝
    gender = content.get('userInfo').get('gender')                           #性别
    urank = content.get('userInfo').get('urank')                             #微博等级
    print("微博昵称：" + name + "\n" + "微博主页地址：" + profile_url + "\n" + "微博头像地址：" + profile_image_url + "\n" + "是否认证：" + str(verified) + "\n" + "微博说明：" + description + "\n" + "关注人数：" + str(guanzhu) + "\n" + "粉丝数：" + str(fensi) + "\n" + "性别：" + gender + "\n" + "微博等级：" + str(urank) + "\n")

if __name__=="__main__":
    get_userInfo(id)
