from sklearn.datasets import load_wine
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical  
from sklearn.preprocessing import OneHotEncoder
from keras.callbacks import EarlyStopping

datasets = load_wine()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size= 0.72, random_state=123, stratify=y)

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#모델 구현
from sklearn.svm import LinearSVC
model = LinearSVC(C = 100)

#컴파일 훈련
history = model.fit(x_train, y_train)

#예측 평가
loss = model.score(x_test, y_test)
y_predict = model.predict(x_test)
print(y_test)
print(y_predict)

acc_score = accuracy_score(y_test, y_predict)
print("정확도 : ", loss)
print("acc score :", acc_score)

'''
기존 : 
정확도 :  0.9800000190734863
============================
ML
정확도 :  0.98
acc score : 0.98
'''