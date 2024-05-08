import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5, 5, 0.1)
def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

y = leaky_relu(x)

plt.plot(x, y)
plt.grid()
plt.show()