import hashlib

m = hashlib.md5()                              #MD5算法加密
m.update(b"Hello")                             #第一段输出值
print(m.hexdigest())
m.update(b"It's me")                           #第二段输出 是两个print拼接输出
print(m.hexdigest())
m.update(b"It's been a long time since we spoken...")

# import hashlib                  各种加密方法
#
# # ######## md5 ########
#
# hash = hashlib.md5()
# hash.update('admin')
# print(hash.hexdigest())
#
# # ######## sha1 ########
#
# hash = hashlib.sha1()
# hash.update('admin')
# print(hash.hexdigest())
#
# # ######## sha256 ########
#
# hash = hashlib.sha256()
# hash.update('admin')
# print(hash.hexdigest())
#
# # ######## sha384 ########
#
# hash = hashlib.sha384()
# hash.update('admin')
# print(hash.hexdigest())
#
# # ######## sha512 ########
#
# hash = hashlib.sha512()
# hash.update('admin')
# print(hash.hexdigest())


