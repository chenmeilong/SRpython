import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# # 2 单变量的线性回归
# 整个2的部分需要根据城市人口数量，预测开小吃店的利润
# 数据在ex1data1.txt里，第一列是城市人口数量，第二列是该城市小吃店利润。

# ## 2.1 Plotting the Data
# 读入数据，然后展示数据
path =  'ex1data1.txt'
data = pd.read_csv(path, header=None, names=['Population', 'Profit'])
print(data.head())
data.plot(kind='scatter', x='Population', y='Profit', figsize=(12,8))
plt.show()

# ## 2.2 梯度下降
# 这个部分你需要在现有数据集上，训练线性回归的参数θ
def computeCost(X, y, theta):
    inner = np.power(((X * theta.T) - y), 2)
    return np.sum(inner) / (2 * len(X))

# ### 2.2.2实现
# 数据前面已经读取完毕，我们要为加入一列x，用于更新θ(0)，然后我们将θ初始化为0，学习率初始化为0.01，迭代次数为1500次

data.insert(0, 'Ones', 1)      #在第0列插入1，列名为’Ones’
# 现在我们来做一些变量初始化。
# 初始化X和y
cols  = data.shape[1]   #cols=3
X = data.iloc[:,:-1]#X是data里的除最后列
y = data.iloc[:,cols-1:cols]#y是data最后一列

# 代价函数是应该是numpy矩阵，所以我们需要转换X和Y，然后才能使用它们。 我们还需要初始化theta。
X = np.matrix(X.values)
y = np.matrix(y.values)
theta = np.matrix(np.array([0,0]))
print(X.shape, theta.shape, y.shape)# 看下维度

# ### 2.2.3计算J(θ)  #这个部分计算J(Ѳ)，X是矩阵
print(computeCost(X, y, theta))    # 计算代价函数 (theta初始值为0)，答案应该是32.07
# ### 2.2.4 梯度下降
# 记住J(θ)的变量是θ，而不是X和y，意思是说，我们变化θ的值来使J(θ)变化，而不是变化X和y的值。
# 一个检查梯度下降是不是在正常运作的方式，是打印出每一步J(θ)的值，看他是不是一直都在减小，并且最后收敛至一个稳定的值。
# θ最后的结果会用来预测小吃店在35000及70000人城市规模的利润。

def gradientDescent(X, y, theta, alpha, iters):
    temp = np.matrix(np.zeros(theta.shape))   #[0. 0.]
    parameters = int(theta.ravel().shape[1])  #需要更新的参数 2   .ravel()扁平化
    cost = np.zeros(iters)
    for i in range(iters):
        error = (X * theta.T) - y
        for j in range(parameters):
            term = np.multiply(error, X[:, j])   # np.multiply 数组对应元素位置相乘
            temp[0, j] = theta[0, j] - ((alpha / len(X)) * np.sum(term))
        theta = temp
        cost[i] = computeCost(X, y, theta)
    return theta, cost

# 这个部分实现了Ѳ的更新
alpha = 0.01   # 初始化一些附加变量 - 学习速率α和要执行的迭代次数，2.2.2中已经提到。
iters = 1500

# 现在让我们运行梯度下降算法来将我们的参数θ适合于训练集。
g, cost = gradientDescent(X, y, theta, alpha, iters)   #g为训练得的模型参数
print(g,cost)
predict1 = [1, 3.5] * g.T
print("predict1:", predict1)
predict2 = [1, 7] * g.T
print("predict2:", predict2)
# 预测35000和70000城市规模的小吃摊利润
x = np.linspace(data.Population.min(), data.Population.max(), 100)  #x轴指定区间取100个等间距点
f = g[0, 0] + (g[0, 1] * x)  #g=[[-3.63029144  1.16636235]]   f为100个x的预测结果
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(x, f, 'r', label='Prediction')  #画线
ax.scatter(data.Population, data.Profit, label='Traning Data') #画点
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')
plt.show()
# 显示原始数据以及拟合的直线




# # 3 多变量线性回归
# ex1data2.txt里的数据，第一列是房屋大小，第二列是卧室数量，第三列是房屋售价
# 根据已有数据，建立模型，预测房屋的售价
path = 'ex1data2.txt'
data2 = pd.read_csv(path, header=None, names=['Size', 'Bedrooms', 'Price'])
print(data2.head())

# ## 3.1 特征归一化
# 观察数据发现，size变量是bedrooms变量的1000倍大小,统一量级会让梯度下降收敛的更快。做法就是，将每类特征减去他的平均值后除以标准差
data2 = (data2 - data2.mean()) / data2.std()
print(data2.head())

# ## 3.2 梯度下降
# 加一列常数项
data2.insert(0, 'Ones', 1)

# 初始化X和y
cols = data2.shape[1]
X2 = data2.iloc[:, 0:cols - 1]
y2 = data2.iloc[:, cols - 1:cols]

# 转换成matrix格式，初始化theta
X2 = np.matrix(X2.values)
y2 = np.matrix(y2.values)
theta2 = np.matrix(np.array([0, 0, 0]))

# 运行梯度下降算法
g2, cost2 = gradientDescent(X2, y2, theta2, alpha, iters)
print(g2)


# ## 3.3 正规方程
# 正规方程是通过求解下面的方程来找出使得代价函数最小的参数的：$\frac{\partial }{\partial {{\theta }_{j}}}J\left( {{\theta }_{j}} \right)=0$ 。
#  假设我们的训练集特征矩阵为 X（包含了${{x}_{0}}=1$）并且我们的训练集结果为向量 y，则利用正规方程解出向量 $\theta ={{\left( {{X}^{T}}X \right)}^{-1}}{{X}^{T}}y$ 。
# 上标T代表矩阵转置，上标-1 代表矩阵的逆。设矩阵$A={{X}^{T}}X$，则：${{\left( {{X}^{T}}X \right)}^{-1}}={{A}^{-1}}$
#
# 梯度下降与正规方程的比较：
#
# 梯度下降：需要选择学习率α，需要多次迭代，当特征数量n大时也能较好适用，适用于各种类型的模型
# 正规方程：不需要选择学习率α，一次计算得出，需要计算${{\left( {{X}^{T}}X \right)}^{-1}}$，如果特征数量n较大则运算代价大，
# 因为矩阵逆的计算时间复杂度为$O(n3)$，通常来说当$n$小于10000 时还是可以接受的，只适用于线性模型，不适合逻辑回归模型等其他模型

# 正规方程
def normalEqn(X, y):
    theta = np.linalg.inv(X.T @ X) @ X.T @ y  # X.T@X等价于X.T.dot(X)
    return theta
final_theta2 = normalEqn(X, y)  # 这里用的是data1的数据
print(final_theta2)
# 梯度下降得到的结果是matrix([[-3.24140214,  1.1272942 ]])