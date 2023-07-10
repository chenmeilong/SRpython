import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family']=['YouYuan']                             #选择幼圆字体

time = np.arange(100)
wheel_circle = np.random.random(100)
GPS_speed = np.random.random(100)*5
wheel_speed = np.random.random(100)*5

fig = plt.figure(figsize=(15,5))   #单位英寸
ax = fig.add_subplot(111)         #绘制单个子图

line1 = ax.plot(time,GPS_speed, '-', label = 'GPS_speed')
line2 = ax.plot(time,wheel_speed, '-', label = 'wheel_speed')
ax2 = ax.twinx()                                               #共享X轴但是Y轴不同
line3 = ax2.plot(time, wheel_circle, '-r', label = 'wheel_circle')
line = line1+line2+line3
labs = [l.get_label() for l in line]
ax.legend(line, labs, loc=0)

ax.grid()
ax.set_xlabel("Time (/100ms)")
ax.set_ylabel("m/s")
ax2.set_ylabel("圈/s")
ax2.set_ylim(-10, 10)
ax.set_ylim(0,10)

plt.show()
