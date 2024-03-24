import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Random seed
np.random.seed(1124)

# create tensor
n_dots = 40
x = np.linspace(0, 1, n_dots)
y = np.sqrt(x) + 0.2*np.random.rand(n_dots) - 0.1

y.shape
x_l = []

for i in range(10):
    x_temp = np.power(x, i+1).reshape(-1, 1)
    x_l.append(x_temp)

X = np.concatenate(x_l, 1)
y.shape


lr = LinearRegression()
lr.fit(X, y)

t = np.linspace(0, 1, 200)
plt.plot(x, y, 'ro', x, lr.predict(X), '-', t, np.sqrt(t), 'r--')
plt.title('10-degree')

plt.savefig('save2.png')
