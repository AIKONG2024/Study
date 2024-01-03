import numpy as np
from keras.models import Sequential
from keras.layers import Dense

#데이터
x = np.array(range(10))
print(x)
print(x.shape) #(3, 10)

# x = x.T
# x = x.transpose()  #벡터로 할때는 필요하지 않음.
print(x)
print(x.shape)

# []안에 들어간 값: list (두개 이상은 리스트)
y = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 
              [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9],
              [9,8,7,6,5,4,3,2,1,0]
              ])
y = y.T


#예측 : [10]

#모델구성
model = Sequential()
model.add(Dense(1, input_dim = 1))
model.add(Dense(5))
model.add(Dense(3))


#컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x,y, epochs=1000, batch_size=1)

#평가, 예측
loss = model.evaluate(x,y)
result = model.predict([10])

print("로스 : ", loss)
print("예측 값: ", result)

#x 값을 배열로 했을 때
# deep : 4(1-1-5-3), epochs =1000, batch_size = 1
# 로스 :  1.3463599809679372e-12
# 예측 값:  [[10.999999   1.9999979 -1.0000007]]

##x 값을 벡터로 했을 때
# deep : 4(1-1-5-3), epochs =1000, batch_size = 1
# 로스 :  2.1482075558514058e-13
# 예측 값:  [[11.         2.        -1.0000005]]