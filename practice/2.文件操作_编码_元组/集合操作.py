
list_1 = [1,4,5,7,3,6,7,9]
list_1 = set(list_1)

list_2 =set([2,6,0,66,22,8,4])
print(list_1,list_2)
'''
#交集
print(  list_1.intersection(list_2) )

#并集
print(list_1.union(list_2))

#差集 in list_1 but not in list_2
print(list_1.difference(list_2))                   #在集合1除去集合2中的数据
print(list_2.difference(list_1))                   #在集合2除去集合1中的数据

list_3 = set([1,3,7])
print(list_3.issubset(list_1))                     #判断是否为子集
print(list_1.issuperset(list_3))                   #判断是否为父集

#对称差集
print(list_1.symmetric_difference(list_2))          #把两个集合都有的去掉


print("-------------")

list_4 = set([5,6,7,8])
print(list_3.isdisjoint(list_4))                  #有交集false 无交集ture
'''

'''
#交集
print(list_1 & list_2)
#并集
print(list_2 | list_1)

#差集
print(list_1 - list_2)      #在集合1除去集合2中的数据

list_3 = set([1,3,7])
print(list_3<=list_1)                     #判断是否为子集
print(list_1>=list_3)                   #判断是否为父集

#对称差集
print(list_1 ^ list_2)         #把两个集合都有的去掉后合并到一个集合
'''
list_1.add(999)                       #集合的增加
list_1.update([888,777,555])          #增加多个
print(list_1)

print(list_1.pop())                   #随机删除一个并打印删除的数据
print(list_1)

print(  list_1.remove(888)  )          #删除不会返回，删除不存在的会报错
print(  list_1.discard(555)  )         #删除不会返回
print(list_1)

