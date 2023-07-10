# 导入 matplotlib 的所有内容（nympy 可以用 np 这个名字来使用）
from pylab import *

# 创建一个 8 * 6 点（point）的图，并设置分辨率为 80
figure(figsize=(8,5), dpi=80)
# 创建一个新的 1 * 1 的子图，接下来的图样绘制在其中的第 1 块（也是唯一的一块）
subplot(1,1,1)
X = np.linspace(-np.pi, np.pi, 256,endpoint=True)
C,S = np.cos(X), np.sin(X)

plot(X, C, color="blue", linewidth=2.5, linestyle="-")              #优化线条
plot(X, S, color="red",  linewidth=2.5, linestyle="-")

xlim(X.min()*1.1, X.max()*1.1)             #边优化
ylim(C.min()*1.1,C.max()*1.1)
# 在屏幕上显示
show()



