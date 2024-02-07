from sklearn.datasets import fetch_covtype
import numpy as np
import pandas as pd
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from sklearn.metrics import accuracy_score 
from sklearn.preprocessing import OneHotEncoder
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint
import time

'''
2    283301
1    211840
3     35754
7     20510
6     17367
5      9493
4      2747
'''

'''
2    96782
1    55097
3    12118
7     4866
6     4577
4      436
5      428
'''

'''
1    83039
0    67909
2    11822
6     7560
5     3346
3      581
4       47
'''

datasets = fetch_covtype()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size= 0.7, random_state=1234, stratify=y)
print(x_test.shape)

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#모델 구성
from sklearn.svm import LinearSVR
model = LinearSVR(C=100)

#컴파일, 훈련
start_time = time.time()
history = model.fit(x_train, y_train)
end_time = time.time()

#평가 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)

acc_score = accuracy_score(y_test, y_predict)
print("acc : ", loss)
print("acc_score : ", acc_score)
#걸린시간 측정 CPU GPU 비교
print("걸린시간 : ", round(end_time - start_time, 2), "초")

'''
기존 : 
loss :  [0.42321649193763733, 0.8228898644447327]
============================
CPU 걸린시간 : 99.98 초
GPU 걸린시간 : 48.25 초
============================
ML
============================
loss : 
'''