#09_1에서 가져옴
import numpy as np
#데이터 가져오기
from sklearn.datasets import load_boston
datasets= load_boston()
x = datasets.data
y = datasets.target

import matplotlib.pyplot as plt

#데이터 분석
print(x.shape)
print(y.shape)

#데이터 전처리
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.7, random_state=20)


#스케일러는 split 후에 해야 x_train의 기준과 동일하게 x_test의 기준을 정해줌.
#predict 할 값도 train 의 기준에 맞춰야함.
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler
# scaler = MinMaxScaler()
scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#데이터 구조 확인
print(x_train.shape)#(301, 13)
print(x_test.shape)#(152, 13)
print(y_train.shape)#(354,)
print(y_test.shape)#(152,)

# CNN 데이터로 변경
# x_train = x_train.reshape(-1, 13, 1, 1) 
# x_test = x_test.reshape(-1, 13, 1, 1)
# print(x_train.shape)

#모델 구성
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, Flatten, LSTM

model = Sequential()
model.add(LSTM(16, input_shape = (13,1)))
model.add(Flatten())
model.add(Dense(20))
model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Dropout(0.3)) 
model.add(Dense(10))
model.add(Dense(1))

from keras.callbacks import ModelCheckpoint
import datetime
date = datetime.datetime.now()
print(date) #2024-01-17 10:52:41.770061
date = date.strftime("%m%d_%H%M")
print(date)

#컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs= 10, batch_size= 10, validation_split=0.7, verbose=1, callbacks= [])

#평가 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)

print('loss : ', loss)
# print('result : ', y_predict)

'''
기존 : 
loss :  369.1446228027344
============================
best : MaxAbs
============================
MinMaxScaler()
 - loss : 209.1923065185547
StandardScaler()
 - loss : 513.5611572265625
MaxAbsScaler()
 - loss : 181.03074645996094
RobustScaler()
 - loss : 410.6644287109375
============================
Dropout() 적용:
loss :  154.19708251953125
============================
CNN 변경
loss :  86.08088684082031
============================
RNN 적용후
loss :  77.0458984375
'''
