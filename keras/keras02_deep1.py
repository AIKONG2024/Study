from keras.models import Sequential
from keras.layers import Dense
import numpy as np 

# 1.데이터
x = np.array([1,2,3])
y = np.array([1,2,3])

# 2.모델구성
model = Sequential()
# model.add(Dense(2, input_dim = 1))
# model.add(Dense(3))#input만 적어도 됨, 간결하게 , hidden layer<- 많다고 좋지않음 예측할수 없어서 hidden layer
# model.add(Dense(10))#이게 윗 줄의 output이 됨.
# model.add(Dense(30))
# model.add(Dense(50))
# model.add(Dense(500))
# model.add(Dense(1500))
# model.add(Dense(5000))
# model.add(Dense(1500))
# model.add(Dense(500))
# model.add(Dense(100))
# model.add(Dense(50))
# model.add(Dense(30))
# model.add(Dense(10))
# model.add(Dense(3))
# model.add(Dense(2))
# model.add(Dense(1))

model.add(Dense(1, input_dim = 1))
model.add(Dense(9000))
model.add(Dense(1))
####결과
# 로스 :  3.254036346334033e-05
# 1/1 [==============================] - 0s 50ms/step
# 예측:  [[4.000371]]

# 3.컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x,y, epochs=100)

# 4. 평가, 예측
loss = model.evaluate(x,y)
print("로스 : ", loss)
result = model.predict([4])
print("예측: ", result)

#epho를 100으로 고정 node 개수, 레이어의 깊이(개수)만 수정
#model.add(Dense(3, input_dim = 1))
# model.add(Dense(6))#input만 적어도 됨, 간결하게 , hidden layer<- 많다고 좋지않음 예측할수 없어서 hidden layer
# model.add(Dense(3))#이게 윗 줄의 output이 됨.
# model.add(Dense(2))#이게 윗 줄의 output이 됨.
# model.add(Dense(1))#이게 윗 줄의 output이 됨.
# ==> [[3.781111]]


