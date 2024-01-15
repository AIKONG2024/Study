#09_1 에서 가져옴

from sklearn.datasets import load_boston

datasets = load_boston()
x = datasets.data #x값
y = datasets.target #y값
# print(x.shape) #(506,13)
# print(y.shape) #(506,)

from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split
import time

#데이터
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.70, random_state= 4291451)


#모델구성
model = Sequential()
# model.add(Dense(64, input_dim = 13))
model.add(Dense(10, input_shape = (13,))) #전체 데이터를 뺀 나머지 데이터를 넣어줌. 행무시. (10000,100,100) -> (100,100)을 넣어줌
model.add(Dense(32))
model.add(Dense(1))

#컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
start_time = time.time()
model.fit(x_train, y_train, epochs=1000, batch_size=16)
end_time = time.time()

#평가, 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)

from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_predict)
print("loss : ", loss)
print("R2 : ", r2)
print("걸린시간 : ", round(end_time - start_time, 2), "초")

