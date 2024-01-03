import numpy as np
from keras.models import Sequential
from keras.layers import Dense

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,6,5,7,8,9,10]) #6,5로 변경한 이유? 

#모델구성
model = Sequential()
model.add(Dense(1, input_dim = 1))
model.add(Dense(5))
model.add(Dense(1))


#컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x,y, epochs=1000, batch_size=1)

#평가, 예측
loss = model.evaluate(x,y)
result = model.predict([110000, 7])

print("로스 : ", loss)
print("예측 값: ", result)

# 로스 :  1.2164491636212915e-11
# 예측 값:  [[1.09999914e+05]
#  [6.99999571e+00]]