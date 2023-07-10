# for i in range(1,10):
#     for j in range(1,i+1):
#         print('{}x{}={}\t'.format(i,j,i*j),end='')
#     print()

#!/usr/bin/python
# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn import metrics

def draw_ROC(y_one_hot,y_score):
    y_one_hot = np.array(y_one_hot, dtype="int32")
    y_score = np.array(y_score, dtype="float64")
    # print(y_score.shape)   #(60, 3)
    print(y_one_hot.dtype)  # int32
    print(y_score.dtype)  # float 64    预测概率矩阵  (60, 3)

    # 首先将矩阵y_one_hot和y_score展开，然后计算假正例率FPR和真正例率TPR
    fpr, tpr, thresholds = metrics.roc_curve(y_one_hot.ravel(), y_score.ravel())  # 就是这个东西   ++上后面的画图就行   传入的是NUMPY数据
    auc = metrics.auc(fpr, tpr)
    print('auc：', auc)  # ROC曲线围成的面积
    # 绘图
    mpl.rcParams['font.sans-serif'] = u'SimHei'  ## 指定默认字体
    mpl.rcParams['axes.unicode_minus'] = False  ## 解决保存图像是负号'-'显示为方块的问题

    # FPR就是横坐标,TPR就是纵坐标
    plt.plot(fpr, tpr, color='green', lw=2, label=u'AUC=%.3f' % auc)
    plt.plot((0, 1), (0, 1), color='gray', lw=1, ls='--')
    plt.xlim((-0.01, 1.02))  # 设置图片现在坐标范围
    plt.ylim((-0.01, 1.02))
    plt.xticks(np.arange(0, 1.1, 0.1))  # 设置刻度
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.xlabel('False Positive Rate', fontsize=13)
    plt.ylabel('True Positive Rate', fontsize=13)
    plt.grid(b=True, ls=':')  # 添加网格线
    plt.legend(loc='lower right', fancybox=True, fontsize=12)  # 设置图例的位置右下角
    plt.title(u'分类后的ROC和AUC', fontsize=17)
    plt.savefig("ROC.png", dpi=72)


if __name__ == "__main__":
    y_one_hot = [[1, 0, 0], [0, 1, 0], [1, 0, 0]]
    y_score = [[0, 0, 0], [0,0,0],
               [1, 0, 0]]  # 其中的和可以不为1

    draw_ROC(y_one_hot,y_score)