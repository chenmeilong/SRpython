#拟合不到 准确的位置

import numpy as np
import matplotlib
matplotlib.use("TkAgg")   #独立窗口
import matplotlib.pyplot as plt

x_data=[338.,333.,328.,207.,226.,25.,179.,60.,208.,606.]
y_data=[640.,633.,619.,393.,428.,27.,193.,66.,226.,1591.]
#y_data = b + w * x_data

x=np.arange(-200,-100,1)  #bias  100个等差值
y=np.arange(-5,5,0.1)    #weight
Z=np.zeros((len(x),len(y)))  #100*100  全是0
X,Y=np.meshgrid(x,y)        #生成网格点坐标矩阵。
for i in range(len(x)):
    for j in range(len(y)):
        b=x[i]
        w=y[j]
        Z[j][i]=0
        for n in range(len(x_data)):   #10次循环
            Z[j][i]=Z[j][i]+(y_data[n]-b-w*x_data[n])**2    #某个点的的   所有数据的方差
        Z[j][i]=Z[j][i]/len(x_data)                     #某个点的的   平均方差

b=-120
w=-4
lr=0.0000001     #调学习率
iteration=100000

b_history=[b]           #-120
w_history=[w]           #-4

for i in range(iteration):
    b_grad=0.0
    w_grad=0.0
    for n in range(len(x_data)):    #10  次求和
        b_grad = b_grad - 2.0 * (y_data[n] - b - w * x_data[n])*1.0         #差值
        w_grad = w_grad - 2.0 * (y_data[n] - b - w * x_data[n]) * x_data[n]  #* x_data[n]  ？？？？？？？

    b = b - lr * b_grad    #权重求和
    w = w - lr * w_grad
    b_history.append(b)
    w_history.append(w)

plt.contourf(x,y,Z,50,alpha=0.5,cmap=plt.get_cmap('jet'))     #画等高线
plt.plot([-188.4],[2.67],'x',ms=12,markeredgewidth=3,color='orange')    #画橙色的x
plt.plot(b_history,w_history,'o-',ms=3,lw=3,color='black')    #画线
plt.xlim(-200,-100)  #坐标
plt.ylim(-5,5)
plt.xlabel(r'$b$',fontsize=16)   #轴标签
plt.xlabel(r'$w$',fontsize=16)
plt.show()


