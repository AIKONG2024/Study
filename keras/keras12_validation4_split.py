from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import pandas as pd

#1. 데이터
x = np.array(range(1, 17))
y = np.array(range(1, 17))

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size= 0.85, shuffle= False ,random_state=66)

print(x_train, y_train)
print(x_test, y_test)

#2. 모델구성
model = Sequential()
model.add(Dense(1, input_dim = 1))
model.add(Dense(5))
model.add(Dense(1))


#컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train,y_train, epochs=1000, batch_size=1,
          validation_split=0.3, #전체 데이터중 30퍼센트를 검증 데이터로 사용하겠다. 통상적으로 evaluate loss > val loss > loss 순으로 신뢰해야함.(훈련에 대한 과적합에 조금 더 자유로움)
          verbose=1
          )

#평가, 예측
loss = model.evaluate(x_test,y_test)
result = model.predict([110000, 7])

print("로스 : ", loss)
print("예측 값: ", result)