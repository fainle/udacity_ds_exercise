import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('data.csv', delimiter=',')  # 加载数据
X = data[:, :-1]  # 变量
y = data[:, -1]  # 值


# 绘制数据点 用于观察一下大致分布
plt.scatter(X, y, marker='.')
plt.show()

# 转换数据集
data = np.hstack((np.ones((data.shape[0], 1)), data))
X_train = data[:, :-1]
y_train = data[:, -1].reshape((-1, 1))


def hypothesis(X, theta):
    """
    进行预测
    """
    return np.dot(X, theta)


# function to compute gradient of error function w.r.t. theta
def gradient(X, y, theta):
    """
    计算梯度
    """
    h = hypothesis(X, theta)
    grad = np.dot(X.transpose(), (h - y))
    return grad


def cost(X, y, theta):
    """
    计算损失函数值
    """
    h = hypothesis(X, theta)
    J = 1 / 2 * np.dot((h - y).transpose(), (h - y))
    return J[0]


def gradient_descent(X, y, learning_rate=0.001, batch_size=25):
    """
    梯度下降算法
    """
    history_cost = []
    theta = np.zeros((X.shape[1], 1))
    n_points = X.shape[0]

    for _ in range(batch_size):
        batch = np.random.choice(range(n_points), batch_size)

        X_batch = X[batch, :]
        y_batch = y[batch]

        theta = theta - learning_rate * gradient(X_batch, y_batch, theta)
        history_cost.append(cost(X_batch, y_batch, theta))

    return theta, history_cost


theta, error_list = gradient_descent(X_train, y_train, batch_size=100)
print("Bias = ", theta[0])
print("Coefficients = ", theta[1:])

# visualising gradient descent
plt.plot(error_list)
plt.xlabel("Number of iterations")
plt.ylabel("Cost")
plt.show()

y_pred = hypothesis(X_train, theta)
#
plt.scatter(X, y, marker='.')
plt.plot(X_train[:, 1], y_pred, color='orange')
plt.show()
