import re    #[]限制范围  {}限制个数     . 任意    * + ? $ ^

#  re.findall  在字符串中找到正则表达式所匹配的所有子串，并把它们作为一个列表返回。


print (re.search(r'Python', 'I love Python').span())   #输出 字符串第一次出现的位置    匹配的位置  元组

print (re.search(r'.', 'I love Python'))   #   . 可以匹配除换行之外的任意字符   匹配了一个I
print (re.search(r'lov.', 'I love Python'))   # . 可以匹配除换行之外的任意字符   匹配了一个I


print (re.search(r'\.', 'I love.Python'))     #反斜杠消除字符特殊意义
print (re.search(r'\d', '123 I love.Python'))  #  匹配数字

print (re.search(r'[aeiouAEIOU]', 'I love.Python'))  # 中括号 限制匹配范围
print (re.search(r'[a-z]', 'I love.Python'))  # 中括号 限制匹配范围
print (re.search(r'[0-2][0-5][0-5]', '123 I love.Python'))   #匹配数字范围
print (re.search(r"[.]", 'I love Python.com'))    #被中括号包含在里面的元字符都会失去特殊功能，就像 反斜杠加上一个元字符是一样的


print  ("*********"*10)
print (re.search(r'ab{3}c', 'abbbc'))        #只能匹配3个b
print (re.search(r'(abc){1,3}', 'abcabcabc12345'))       #匹配1-3个abc
print (re.search(r'b{3,5}', 'abbbbc'))        #  匹配3-5个b

######综合应用  匹配IP地址
print (re.search('[01]\d\d|2[0-4]\d|25[0-5]', '188'))  #匹配0-255     | 表示或

print (re.search('(([01]{0,1}\d{0,1}\d|2[0-4]\d|25[0-5])\.){3}[01]{0,1}\d{0,1}\d|2[0-4]\d|25[0-5]', '192.168.42.1'))   #匹配IP地址
print  ("*********"*10)


#托字符(^)：定位匹配，匹配字符串的开始位置（即确定一个位置）。
print ( re.search(r"^Python", "Python,I love "))
#跟托字符（^）对应的就是 美元符号（$），$ 匹配输入字符串的结束位置.
print (re.search(r"Python$", "I love Python"))
print (re.search(r"en+","ennnnnnnn"))    #+表示一次或者多次   *表示0次或者多次   ？表示0次或者1次
print (re.search(r"en*","ennnnnnnn"))    #+表示一次或者多次   *表示0次或者多次   ？表示0次或者1次
print (re.search(r"en?","ennnnnnnn"))    #+表示一次或者多次   *表示0次或者多次   ？表示0次或者1次
print (re.search(r"en{3,5}?","ennnnnnnn"))    #匹配 3次  尽可能少      去掉？号 就是尽可能多
print (re.search(r"en{3,5}?","ennnnnnnn"))    #匹配 3次  尽可能少      去掉？号 就是尽可能多  即非贪婪模式
print  ("*********"*10)


print (re.search(r"python(C|D)","12345pythonC"))     #|表示与
print  (re.findall(r"[^a-z]","Python.com"))      #寻找所有的^表示取反

print (re.search('<.+?>',"<html><title>"))     #非贪婪模式
print (re.search('<.+>',"<html><title>"))     #贪婪模式

