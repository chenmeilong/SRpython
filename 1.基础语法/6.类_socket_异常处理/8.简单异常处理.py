names = ['alex','jack']
data = {}

# try:
#     names[3]
#     data['name']
# except KeyError as e :
#     print("没有这个key",e)
# except IndexError as e :                    #错误名字
#     print("列表操作错误",e)

# try:
#     names[3]
#     data['name']
# except (KeyError,IndexError)  as e :                    #多个错误一起写
#     print("列表操作错误",e)

# try:
#     open("tes.txt")
#     names[3]
#     data['name']
# except Exception  as e :                    #全部错误一起写   所有错误都能抓  但是不清楚哪出错了
#     print("出错了",e)


# try:                                                #未知错误   代码格式错误  语法错误 抓不了
#     open("tes.txt")
# except (KeyError,IndexError) as e :
#     print("没有这个key",e)
# except IndexError as e :
#     print("列表操作错误",e)
# except Exception as e:
#     print("未知错误", e)


try:
     a=1
     print(a)
except (KeyError,IndexError) as e :
    print("没有这个key",e)
except IndexError as e :
    print("列表操作错误",e)
except BaseException as e:
    print("未知错误",e)
else:                                                 #一切正常执行这个
    print("一切正常")
finally:
    print("不管有没有错，都执行")                           #不管有没有错，都执行
