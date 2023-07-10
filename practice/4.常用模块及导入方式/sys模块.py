import sys

# sys.argv                               命令行参数List，第一个元素是程序本身路径
# sys.exit(n)                            退出程序，正常退出时exit(0)
# sys.version                            获取Python解释程序的版本信息
# sys.maxint                             最大的Int值
# sys.path                               返回模块的搜索路径，初始化时使用PYTHONPATH环境变量的值
# sys.platform                           返回操作系统平台名称
# sys.stdout.write('please:')            不跨行输入
# val = sys.stdin.readline()[:-1]

# import sys
# sys.path.append('D:\Wayne\Desktop/alibaba\FCN\yolov3_master')   #这样增加系统路径 可以在这个路径下找到自己需要导入的模块 常用
# from models import Darknet

print(sys.argv)

# import os                  #将上级目录添加到环境变量
# sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))
# print(sys.path)

from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
print(ROOT)
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = ROOT.relative_to(Path.cwd())  # relative
print(ROOT)

