
product_list = [
    ('Iphone',5800),
    ('Mac Pro',9800),
    ('Bike',800),
    ('Watch',10600),
    ('Coffee',31),
    ('Alex Python',120),
]
#shopping_list = [5000]
salary = input("Input your salary:")
if salary.isdigit():                                         #检测字符串是否是数字组成
    salary = int(salary)                                     #将input字符型转整型
    while True:                                              #死循环
        for index,item in enumerate(product_list):           #enumerate 是索引序号操作 语法  内置方法会讲到
        #for item in product_list:                                                #另外一种写法
            #print(product_list.index(item),item)                                 #续
            print(index,item)
        user_choice = input("选择要买嘛？>>>:")
        if user_choice.isdigit():
            user_choice = int(user_choice)
            if user_choice < len(product_list) and user_choice >=0:
                p_item = product_list[user_choice]
                if p_item[1] <= salary:                       #买的起
                    shopping_list.append(p_item)
                    salary -= p_item[1]
                    print("Added %s into shopping cart,your current balance is \033[31;1m%s\033[0m" %(p_item,salary) )      #添加颜色\033[31;1m   \033[0m
                else:
                    print("\033[41;1m你的余额只剩[%s]啦，还买个毛线\033[0m" % salary)
            else:
                print("product code [%s] is not exist!"% user_choice)
        elif user_choice == 'q':
            print("--------shopping list------")
            for i in shopping_list:                                              #循环列表
                print(i)
            print("Your current balance:",salary)
            exit()                                                              #退出死循环
        else:
            print("invalid option")