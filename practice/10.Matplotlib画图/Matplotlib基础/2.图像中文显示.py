import numpy as np
from matplotlib import pyplot as plt
import matplotlib

a=sorted([f.name for f in matplotlib.font_manager.fontManager.ttflist])        #输出系统字体
for i in a:
    print(i)
plt.rcParams['font.family']=['YouYuan']                             #选择幼圆字体


x = np.arange(1, 11)
y = 2 * x + 5
plt.title("测试")
# fontproperties 设置中文显示，fontsize 设置字体大小
plt.xlabel("x 轴")
plt.ylabel("y 轴",fontsize=20)
plt.plot(x, y)
plt.show()
