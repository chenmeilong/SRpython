#key-value                     字典是无序的


'''
info = {
    'stu1101': "TengLan Wu",
    'stu1102': "LongZe Luola",
    'stu1103': "XiaoZe Maliya",
}

print(info.get('stu1103'))                     #查找   有就获取输出，无就输出none        常用查找
print('stu1103' in info)                       #查找  ture or false
'''
'''
#print(info["stu1101"])                       #查找
info["stu1101"] ="武藤兰"                     #修改
info["stu1104"] ="CangJingkong"               #增加

#del info["stu1101"]                    #删除法1
info.pop("stu1101")                     #删除法2
info.popitem()                          #随机删除
print(info)
'''
'''
av_catalog = {                                #目录
    "欧美":{
        "www.youporn.com": ["很多免费的,世界最大的","质量一般"],
        "www.pornhub.com": ["很多免费的,也很大","质量比yourporn高点"],
        "letmedothistoyou.com": ["多是自拍,高质量图片很多","资源不多,更新慢"],
        "x-art.com":["质量很高,真的很高","全部收费,屌比请绕过"]
    },
    "日韩":{
        "tokyo-hot":["质量怎样不清楚,个人已经不喜欢日韩范了","听说是收费的"]
    },
    "大陆":{
        "1024":["全部免费,真好,好人一生平安","服务器在国外,慢"]
    }
}
av_catalog["大陆"]["1024"][1] = "可以在国内做镜像"

av_catalog.setdefault("台湾",{"www.baidu.com":[1,2]})
print(av_catalog)
'''

'''
info = {
    'stu1101': "TengLan Wu",                         #keey  value
    'stu1102': "LongZe Luola",
    'stu1103': "XiaoZe Maliya",
}

b ={
    'stu1101': "Alex",
    1:3,
    2:5
}

info.update(b)                                       #两个字典合拼，有相同的覆盖，无相同的创建
print(info )
#print(info.items() )                                 #字典转列表
c = dict.fromkeys([6,7,8],[1,{"name":"alex"},444])    #生成 keey  但是后面的内容不能改，
print(c )
c[7][1]['name'] = "Jack Chen"
print(c)

'''


info = {
    'stu1101': "TengLan Wu",                         #keey  value
    'stu1102': "LongZe Luola",
    'stu1103': "XiaoZe Maliya",
}

for i in info:                                         #常用的循环字典  推荐使用
    print(i,info[i])

for k,v in info.items():                                #运行效率低于上面的
    print(k,v)

