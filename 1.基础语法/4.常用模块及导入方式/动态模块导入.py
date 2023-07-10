import importlib

# module2=__import__('module2.module_chen')  # 这是解释器自己内部用的    不建议用
# print(module2.module_chen.name)

a='module2.module_chen'
module2=importlib.import_module(a)     #与上面这句效果一样，官方建议用这个
print(module2.name)                                       #注意  这里很特殊


