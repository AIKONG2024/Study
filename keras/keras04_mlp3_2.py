import numpy as np
from keras.models import Sequential
from keras.layers import Dense

#데이터
x = np.array([range(10), range(21, 31), range(201, 211)])
print(x)
print(x.shape) #(3, 10)

# x = x.T
x = x.transpose()
print(x)
print(x.shape)

# []안에 들어간 값: list (두개 이상은 리스트)
y = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 
              [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9],
              [9,8,7,6,5,4,3,2,1,0]
              ])
y = y.T


#모델 구성
model = Sequential()
model.add(Dense(1, input_dim = 3))
model.add(Dense(5))
model.add(Dense(3))

#컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x,y, epochs=2000, batch_size=1)


#평가, 예측
loss = model.evaluate(x,y)
result = model.predict([[10,31,211]])

print('로스값 : ', loss)
print('예측값 : ', result)

#dense deep =4 (3-1-5-3), epochs = 2000, batch_size = 1
# 로스값 :  3.8381152639779814e-13
# 예측값 :  [[10.999999   2.0000005 -1.0000012]]
