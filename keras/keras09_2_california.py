from sklearn.datasets import fetch_california_housing

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

print(x)
print(y)
print(x.shape , y.shape) #(20640, 8) (20640,)

print(datasets.feature_names) #['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
print(datasets.DESCR)
'''
:Attribute Information:
        - MedInc        median income in block group
        - HouseAge      median house age in block group
        - AveRooms      average number of rooms per household
        - AveBedrms     average number of bedrooms per household
        - Population    block group population
        - AveOccup      average number of household members
        - Latitude      block group latitude
        - Longitude     block group longitude
'''

#[실습]
# R2 0.55 ~ 0.6 이상
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split
import time


#데이터
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.75,random_state=9211111)

#모델 구성
model = Sequential()
model.add(Dense(64, input_dim = 8))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(1))

#컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
start_time = time.time()
model.fit(x_train, y_train, epochs=1000, batch_size=300)
end_time = time.time()


#평가, 예측
from sklearn.metrics import r2_score
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)

print("loss : ", loss)
print("R2 : ", r2)
print("걸린시간 : ", round(end_time - start_time, 2), "초")

#train_size : 0.7 /  deep 3 (13-64-64-1)  / random_state : 54 / epochs = 5 / batch_size = 1

