from keras.models import Sequential
from keras.layers import Dense
import numpy as np 

#과제1) epho를 100으로 고정 node 개수, 레이어의 깊이(개수)만 수정
#4.000x만들기
#3.999x만들기

# 1.데이터
x = np.array([1,2,3])
y = np.array([1,2,3])

# 2.모델구성
model = Sequential()

model.add(Dense(1, input_dim = 1))
model.add(Dense(9000))
model.add(Dense(1))

# 3.컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x,y, epochs=100)

# 4. 평가, 예측
loss = model.evaluate(x,y)
print("로스 : ", loss)
result = model.predict([4])
print("예측: ", result)

#======================================================================================
# 결과
# 로스 :  3.254036346334033e-05
# 1/1 [==============================] - 0s 50ms/step
# 예측:  [[4.000371]]
####===================================================================================


