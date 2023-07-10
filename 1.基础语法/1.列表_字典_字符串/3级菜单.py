data = {
    '北京':{
        "昌平":{
            "沙河":["oldboy","test"],
            "天通苑":["链家地产","我爱我家"]
        },
        "朝阳":{
            "望京":["奔驰","陌陌"],
            "国贸":{"CICC","HP"},
            "东直门":{"Advent","飞信"},
        },
        "海淀":{},
    },
    '山东':{
        "德州":{},
        "青岛":{},
        "济南":{}
    },
    '广东':{
        "东莞":{},
        "常熟":{},
        "佛山":{},
    },
}
exit_flag = False

while not exit_flag:                                                 #死循环1
    for i in data:                                                      #显示一级菜单
        print(i)
    choice = input("选择进入1>>:")
    if choice in data:
        while not exit_flag:                                         #死循环2
            for i2 in data[choice]:                                       #显示二级菜单
                print("\t",i2)
            choice2 = input("选择进入2>>:")
            if choice2 in data[choice]:
                while not exit_flag:                                #死循环3
                    for i3 in data[choice][choice2]:                         #显示三级菜单
                        print("\t\t", i3)
                    choice3 = input("选择进入3>>:")
                    if choice3 in data[choice][choice2]:                       #修复没有在菜单中会出现的bug
                        for i4 in data[choice][choice2][choice3]:                 #进入三级菜单
                            print("\t\t\t",i4)
                        choice4 = input("最后一层，按b返回>>:")
                        if choice4 == "b":
                            pass              #保证程序结构完整性，没有什么意义     退出到前面的死循环3
                        elif choice4 == "q":
                            exit_flag = True                                       #不错的思路
                    if choice3 == "b":
                        break
                    elif choice3 == "q":
                        exit_flag = True
            if choice2 == "b":
                break
            elif choice2 == "q":
                exit_flag = True