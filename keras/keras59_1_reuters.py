from keras.datasets import reuters
import numpy as np
import pandas as pd

(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=10,
                                                         test_split=0.2)
# print(x_train)
print(x_train.shape, x_test.shape)#(8982,) (2246,)
print(y_train.shape, y_test.shape)#(8982,) (2246,)
print(type(x_train))#<class 'numpy.ndarray'>
print(y_train) #[ 3  4  3 ... 25  3 25]
unique, count = np.unique(y_train, return_counts=True)
print(unique, count, len(unique))
print(type(x_train[0]))#<class 'list'>
print(len(x_train[0]))
print(len(x_train[1]))

print("뉴스기사의 최대길이 : ", max(len(i) for i in x_train)) #2376
print("뉴스기사의 평균길이 : ", sum(map(len, x_train))/len(x_train)) #145.5398574927633

#전처리
from keras.utils import pad_sequences
x_train = pad_sequences(x_train, padding='pre', maxlen=100, 
                        truncating='pre')

x_test = pad_sequences(x_train, padding='pre', maxlen=100, 
                        truncating='pre')


print(x_train.shape, y_train.shape) #(8982, 100) (8982,)