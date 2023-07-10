import re


#反斜杠后面加的是数字
#①如果跟着的数字是 1~99，就表示引用序号对应的值组所匹配的字符串，其中序号所对应的值组：为 \ 前面的值组，\序号必须在对应的值组的正后面，序号为第几个值组。
print (re.search(r"(Python)\1", "I love PythonPython"))      #  (Python)是第一个值组     r'(Python)\1' 就等于 'PythonPython'。
print (re.search(r"(love)\1(Python)\2", "I lovelovePythonPython"))
print (re.search(r"(I )love(Python)\2", "I lovePythonPython.com"))
#②如果跟着的数字是 0 或者 3位的数字，那么它是一个八进制数，表示的是这个八进制数对应的 ASCII 码对应的字符。
print (re.search(r"I love Python\060", 'I love Python0'))    #0的asll码是060

#反斜杠后面加的是字母    \d 匹配为数字 \D匹配的与\d相反   \s 匹配包括[ \t\n\r\f\v] 以及其他空白字符
print ("反斜杠后面加的是字母****************")
print (re.findall(r"\bFishC\b", "FishC.com!FishC_com!(FishC)")) #\b 也是一个临框断言，它是匹配一个单词的边界， 匹配两边都不是数字字母下划线的
print (re.findall(r"\bFishC\B", "FishC.com!FishC_com!(FishC)"))  #与\B相反
print (re.findall(r"\w", "我爱Python3 (love_python.com!)"))   #\w 所有语言的字符数字下划线



