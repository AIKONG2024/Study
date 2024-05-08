import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5, 5, 0.1)

def selu(x):
    alpha = 1.67326324
    scale = 1.05070098
    return scale * np.where(x > 0, x, alpha * (np.exp(x) - 1))
y = selu(x)
plt.plot(x, y)
plt.grid()
plt.show()