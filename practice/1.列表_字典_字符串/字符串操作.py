name = "my \tname is {name} and i am {year} old"

print(name.capitalize())                                    #首字母大写
print(name.count("a"))                                      #找个数
print(name.center(50,"-"))                                  #打印出来居中补足50个个-凑数
print(name.endswith("ex"))                                  #判断结尾是输出True 否输出False
print(name.expandtabs(tabsize=30))                          #转意占位符\t           不常用
print("???")
print(name.find("z"))
print(name[name.find("name"):])                             #查找字符开始位置往后打印出来
print(name.format(name='alex',year=23))                     #{}站位输出
print(name.format_map(  {'name':'alex','year':12}  ))       #字典                    不常用
print('ab23'.isalnum())                                     #是否为英文或数字   特殊字符会false
print('abA'.isalpha())                                      #是否为纯英文字符 无论大小写
print('1A'.isdigit())                                    #是否为整数         ********************
print('a 1A'.isidentifier())                                #判读是不是一个合法的标识符  也就是数字字母下划线   不常用
print('33'.isnumeric())                                    #只能有数字                不常用
print('My Name Is  '.istitle())                            #是否每个单词第一个都是大写
print('My Name Is  '.isprintable())                        #tty file ,drive file 才用       不常用
print('My Name Is  '.isupper())                            #是否全是大写
print('+'.join( ['1','2','3'])  )                          #连接列表    \\\\\\\\\\\\\\\\\\\\\\常用
print( name.ljust(50,'*')  )                               #不够后面补起来
print( name.rjust(50,'-')  )                               #不够前面补起来
print( 'Alex'.lower()  )                                   #全小写
print( 'Alex'.upper()  )                                   #全大写
print( '\nAlex'.lstrip()  )                                #去除左边的空格和回车 \\\\\\\\\\\\\\常用
print( 'Alex\n'.rstrip()  )                                #去除右边的空格和回车 \\\\\\\\\\\\\\常用
print( '    Alex\n'.strip()  )                             #去除两边的空格和回车 \\\\\\\\\\\\\\常用

p = str.maketrans("abcdefli",'123$@456')                   #替换  相应        不常用
print("alex li".translate(p) )

print('alex li'.replace('l','L',1))                        #替换l换L换一个   \\\\\\\\\\\\\\\\\常用
print('alex lil'.rfind('l'))                               #从左往右找，输出最右边的一个位置
print('1+2+3+4'.split('+'))                               #分离字符串到列表
print('1+2\n+3+4'.splitlines())                           #按换行符分列表
print('Alex Li'.swapcase())                               #大小写互换
print('lex li'.title())                                   #每个单词第一个大写变成标题
print('lex li'.zfill(50))                                 #补零

print( '---')

