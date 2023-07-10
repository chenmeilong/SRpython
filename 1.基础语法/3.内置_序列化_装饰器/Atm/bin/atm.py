
import os
import sys
BASE_DIR = os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) )      #返回目录名不要文件名    一层一层返回

sys.path.append(BASE_DIR)                                                      #添加环境变量       动态添加所以标红
from conf import settings                                                  #调用模块
from core import main

main.login()                                                                #调用其他模块函数
