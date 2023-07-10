import re

#需要重复地使用某个正则表表达式，可以先把该正则表达式编译成模式对象。
p = re.compile(r"[A-Z]")
print (p.search("I love Python"))
print (p.findall("I love Python"))

#编译标志
# ASCII, A	使得转义符号如 \w，\b，\s 和 \d 只能匹配 ASCII 字符
# DOTALL, S	使得 . 匹配任何符号，包括换行符
# IGNORECASE, I	匹配的时候不区分大小写
# LOCALE, L	支持当前的语言（区域）设置
# MULTILINE, M	多行匹配，影响 ^ 和 $
# VERBOSE, X (for 'extended')	启用详细的正则表达式


# finditer() ，是将结果返回一个迭代器，方便以迭代方式获取数据。
# sub() ，是实现替换的操作。





