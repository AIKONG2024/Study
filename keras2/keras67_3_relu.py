import  numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5, 5, 0.1)

# def relu(x):
#     return np.maximum(x, 0)

relu = lambda x: np.maximum(x, 0)
y = relu(x)
plt.plot(x, y)
plt.grid()
plt.show()