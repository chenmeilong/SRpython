# ## 协同过滤
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.io import loadmat
# 推荐引擎使用基于项目和用户的相似性度量来检查用户的历史偏好，以便为用户可能感兴趣的新“事物”提供建议。
# 在本练习中，我们将实现一种称为协作过滤的特定推荐系统算法，并将其应用于 电影评分的数据集。
data = loadmat('ex8_movies.mat')
print(data)

# Y是包含从1到5的等级的（数量的电影x数量的用户）数组.R是包含指示用户是否给电影评分的二进制值的“指示符”数组。 两者应该具有相同的维度。
Y = data['Y']
R = data['R']
print(Y.shape, R.shape)   #(1682, 943) (1682, 943)

# 我们可以通过平均排序Y来评估电影的平均评级。
print(Y[1, np.where(R[1, :] == 1)[0]].mean())   #3.2061068702290076

# 我们还可以通过将矩阵渲染成图像来尝试“可视化”数据。 我们不能从这里收集太多，但它确实给我们了解用户和电影的相对密度。
fig, ax = plt.subplots(figsize=(12, 12))
ax.imshow(Y)
ax.set_xlabel('Users')
ax.set_ylabel('Movies')
fig.tight_layout()
plt.show()

# 接下来，我们将实施协同过滤的代价函数。 直觉上，“代价”是指一组电影评级预测偏离真实预测的程度。
# 代价方程在练习文本中给出。 它基于文本中称为X和Theta的两组参数矩阵。 这些“展开”到“参数”输入中，以便稍后可以使用SciPy的优化包。
# 请注意，我已经在注释中包含数组/矩阵形状（对于我们在本练习中使用的数据），以帮助说明矩阵交互如何工作。

def serialize(X, theta):
    """序列化两个矩阵
    """
    # X (movie, feature), (1682, 10): movie features
    # theta (user, feature), (943, 10): user preference
    return np.concatenate((X.ravel(), theta.ravel()))


def deserialize(param, n_movie, n_user, n_features):
    """逆序列化"""
    return param[:n_movie * n_features].reshape(n_movie, n_features), param[n_movie * n_features:].reshape(n_user,
                                                                                                           n_features)
# recommendation fn
def cost(param, Y, R, n_features):
    """compute cost for every r(i, j)=1
    Args:
        param: serialized X, theta
        Y (movie, user), (1682, 943): (movie, user) rating
        R (movie, user), (1682, 943): (movie, user) has rating
    """
    # theta (user, feature), (943, 10): user preference
    # X (movie, feature), (1682, 10): movie features
    n_movie, n_user = Y.shape
    X, theta = deserialize(param, n_movie, n_user, n_features)
    inner = np.multiply(X @ theta.T - Y, R)
    return np.power(inner, 2).sum() / 2

# 为了测试这一点，我们提供了一组我们可以评估的预训练参数。 为了保持评估时间的少点，我们将只看一小段数据。
params_data = loadmat('ex8_movieParams.mat')
X = params_data['X']
theta = params_data['Theta']
print(X.shape, theta.shape)   #(1682, 10) (943, 10)

users = 4
movies = 5
features = 3
X_sub = X[:movies, :features]
theta_sub = theta[:users, :features]
Y_sub = Y[:movies, :users]
R_sub = R[:movies, :users]
param_sub = serialize(X_sub, theta_sub)
print(cost(param_sub, Y_sub, R_sub, features))  #22.224603725685675

param = serialize(X, theta)  # total real params
print(cost(serialize(X, theta), Y, R, 10))  # this is real total cost   27918.64012454421

# # gradient

#  接下来我们需要实现梯度计算。 就像我们在练习4中使用神经网络实现一样，我们将扩展代价函数来计算梯度。
def gradient(param, Y, R, n_features):
    # theta (user, feature), (943, 10): user preference
    # X (movie, feature), (1682, 10): movie features
    n_movies, n_user = Y.shape
    X, theta = deserialize(param, n_movies, n_user, n_features)
    inner = np.multiply(X @ theta.T - Y, R)  # (1682, 943)
    # X_grad (1682, 10)
    X_grad = inner @ theta
    # theta_grad (943, 10)
    theta_grad = inner.T @ X
    # roll them together and return
    return serialize(X_grad, theta_grad)

n_movie, n_user = Y.shape
X_grad, theta_grad = deserialize(gradient(param, Y, R, 10), n_movie, n_user, 10)
print(X_grad, theta_grad)

# 我们的下一步是在代价和梯度计算中添加正则化。 我们将创建一个最终的正则化版本的功能（请注意，此版本包含一个额外的“学习率”参数，在文本中称为“lambda”）。

def regularized_cost(param, Y, R, n_features, l=1):
    reg_term = np.power(param, 2).sum() * (l / 2)
    return cost(param, Y, R, n_features) + reg_term

def regularized_gradient(param, Y, R, n_features, l=1):
    grad = gradient(param, Y, R, n_features)
    reg_term = l * param
    return grad + reg_term

print(regularized_cost(param_sub, Y_sub, R_sub, features, l=1.5))  #31.34405624427422
print(regularized_cost(param, Y, R, 10, l=1))  # total regularized cost   #32520.682450229557

n_movie, n_user = Y.shape
X_grad, theta_grad = deserialize(regularized_gradient(param, Y, R, 10),
                                 n_movie, n_user, 10)

# 这个结果再次与练习代码的预期输出相匹配，所以看起来正则化是正常的。 在我们训练模型之前，我们有一个最后步骤，
# 我们的任务是创建自己的电影评分，以便我们可以使用该模型来生成个性化的推荐。 为我们提供一个连接电影索引到其标题的文件。 接着我们将文件加载到字典中。

movie_list = []
f = open('movie_ids.txt', encoding='gbk')
for line in f:
    tokens = line.strip().split(' ')
    movie_list.append(' '.join(tokens[1:]))
movie_list = np.array(movie_list)
print(movie_list[0])

# 我们将使用练习中提供的评分。
ratings = np.zeros((1682, 1))
ratings[0] = 4
ratings[6] = 3
ratings[11] = 5
ratings[53] = 4
ratings[63] = 5
ratings[65] = 3
ratings[68] = 5
ratings[97] = 2
ratings[182] = 4
ratings[225] = 5
ratings[354] = 5
print(ratings.shape)

# 我们可以将自己的评级向量添加到现有数据集中以包含在模型中。

Y = data['Y']
Y = np.append(ratings, Y, axis=1)  # now I become user 0
print(Y.shape)

R = data['R']
R = np.append(ratings != 0, R, axis=1)
print(R.shape)

# 我们不只是准备训练协同过滤模型。 我们只需要定义一些变量并对评级进行规一化。

movies = Y.shape[0]  # 1682
users = Y.shape[1]  # 944
features = 10
learning_rate = 10.

X = np.random.random(size=(movies, features))
theta = np.random.random(size=(users, features))
params = serialize(X, theta)

print(X.shape, theta.shape, params.shape)

Y_norm = Y - Y.mean()
print(Y_norm.mean())

# # training
from scipy.optimize import minimize
fmin = minimize(fun=regularized_cost, x0=params, args=(Y_norm, R, features, learning_rate),
                method='TNC', jac=regularized_gradient)
print(fmin)

# 我们训练好的参数是X和Theta。 我们可以使用这些来为我们添加的用户创建一些建议。
X_trained, theta_trained = deserialize(fmin.x, movies, users, features)
print(X_trained.shape, theta_trained.shape)

# 最后，使用训练出的数据给出推荐电影
prediction = X_trained @ theta_trained.T
my_preds = prediction[:, 0] + Y.mean()
idx = np.argsort(my_preds)[::-1]  # Descending order
print(idx.shape)
print(my_preds[idx][:10])
for m in movie_list[idx][:10]:
    print(m)
