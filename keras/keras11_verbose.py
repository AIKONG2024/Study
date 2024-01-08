#06_1 카피

import numpy as np
from keras.models import Sequential
from keras.layers import Dense

#1. 데이터
# x = np.array([1,2,3,4,5,6,7,8,9,10])
# y = np.array([1,2,3,4,6,5,7,8,9,10])

#60~90 사이로 분리 많이 함.
x_train = np.array([1,2,3,4,5]) #전체 데이터의 70%로 평가
y_train = np.array([1,2,3,4,6])

#train에서 또 나뉜것.
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
model.fit(x_train,y_train, epochs=100, batch_size=1,
          verbose=100)
#verbose=0 : 침묵
#verbose=1 : 디폴트
#verbose=2 : 프로그래스바 삭제(큰데이터는 디폴트가 좋음. 확인 필수)
#verbose=3 : 에포만 나옴
#verbose=나머지 : 에포만 나옴

#평가, 예측
loss = model.evaluate(x_test,y_test)
result = model.predict([110000, 7])

print("로스 : ", loss)
print("예측 값: ", result)

#데이터 쪼개기 전
# 로스 :  1.2164491636212915e-11
# 예측 값:  [[1.09999914e+05]
#  [6.99999571e+00]]

#데이터 쪼갠 후
# 로스 :  6.821210263296962e-13
# 예측 값:  [[1.09999984e+05]
#  [6.99999905e+00]]

"""
    터미널에서 결과가 7/7 [==============================] - 0s 3ms/step - loss: 0.2834 는 훈련 값으로 돌린 로스 값,
    1/1 [==============================] - 0s 115ms/step - loss: 0.0301 의 결과값은 평가 값으로 돌린 로스 값.
"""