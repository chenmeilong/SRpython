import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family']=['YouYuan']                             #选择幼圆字体

plt.title("速度变化曲线图")
plt.xlabel("x 轴")
plt.ylabel("y 轴",fontsize=10)

t_list = []
result_list = []
t = 0
while True:
    if t >= 50:
        t = 0
        plt.clf()
        t_list.clear()
        result_list.clear()
    else:
        t +=1
        t_list.append(t)
        result_list.append(0)
        plt.plot(t_list, result_list, c='r', ls='-', marker='o', mec='b', mfc='w',color='b')  ## 保存历史数据
        plt.pause(0.1)

