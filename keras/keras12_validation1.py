#06_1 카피

import numpy as np
from keras.models import Sequential
from keras.layers import Dense

#60~90 사이로 분리 많이 함.
x_train = np.array([1,2,3,4,5]) 
y_train = np.array([1,2,3,4,6])

#train에서 또 나뉜것. #검증 비율은 알아서 하지만 8:2 7:3,6:4 정보 
x_val = np.array([6,7])
y_val = np.array([5,7])


x_test = np.array([8,9,10]) #전체 데이터의 30%로 평가 #소멸하는 데이터
y_test = np.array([8,9,10]) #자른데이터와 평가한 데이터의 결과가 잘 나오면 신뢰가 더 오름.

#모델구성
model = Sequential()
model.add(Dense(1, input_dim = 1))
model.add(Dense(5))
model.add(Dense(1))


#컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train,y_train, epochs=1000, batch_size=1,
          validation_data=(x_val, y_val))

#평가, 예측
loss = model.evaluate(x_test,y_test)
result = model.predict([110000, 7])

print("로스 : ", loss)
print("예측 값: ", result)