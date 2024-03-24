import numpy as np
import matplotlib.pyplot as plt

# Random seed
np.random.seed(1124)

# create tensor
n_dots = 40
x = np.linspace(0, 1, n_dots)
y = np.sqrt(x) + 0.2*np.random.rand(n_dots) - 0.1

def plot_polynomial_fit(x, y, deg):
    # Polynomial fitting is performed on the data
    p = np.poly1d(np.polyfit(x, y, deg))

    t = np.linspace(0, 1, 200)

    # Draw raw data (red dots), fit results (blue solid line), and ideal results (red dashed line)
    plt.plot(x, y, 'ro', label='Original Data')
    plt.plot(t, p(t), '-', label=f'Degree {deg} Fit')
    plt.plot(t, np.sqrt(t), 'r--', label='Ideal Result')

    # 显示图例
    plt.legend()

plt.figure(figsize=(18, 4), dpi=200)
degrees = [1, 3, 10]
titles = ['Under Fitting', 'Fitting', 'Over Fitting']
for index, deg in enumerate(degrees):
    plt.subplot(1, 3, index + 1)
    plot_polynomial_fit(x, y, deg)
    plt.title(titles[index], fontsize=20)

plt.savefig('save1.png')
