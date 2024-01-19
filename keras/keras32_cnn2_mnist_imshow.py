import numpy as np
from keras.datasets import mnist
import pandas as pd

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, y_train.shape)#(60000, 28, 28) : ==> 흑백 (60000, 28,28, 1)인데 생략 //(60000,)
print(x_test.shape, y_test.shape)#(10000, 28, 28) //(10000,) 
print(x_train[0])
unique, count = np.unique(y_train, return_counts=True)
print(unique, count) #[0 1 2 3 4 5 6 7 8 9] [5923 6742 5958 6131 5842 5421 5918 6265 5851 5949]
print(pd.value_counts(y_test))
'''
1    1135
2    1032
7    1028
3    1010
9    1009
4     982
0     980
8     974
6     958
5     892
'''

import matplotlib.pyplot as plt
plt.imshow(x_train[59999], 'gray')
plt.show()