import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5, 5, 0.1)

mish = lambda x: x * (np.tanh(np.log(1+np.exp(x)))) #x * tahn(softplus(x))  // #softplus(x) = log(1 + exp(x))
y = mish(x)
plt.plot(x, y)
plt.grid()
plt.show()