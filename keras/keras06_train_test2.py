import numpy as np
from keras.models import Sequential
from keras.layers import Dense

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,6,5,7,8,9,10])

#[실습] 넘파이 리스트의 슬라이싱!! 7:3으로 잘라!!
# x_train = x[:7] 
# y_train = y[:7]
# x_train = x[0:7] 
# y_train = y[0:7]
# x_train = x[:-3]
# y_train = y[:-3]

x_train = x[0:7:1] #[1 2 3 4 5 6 7]
y_train = y[0:7:1] #[1 2 3 4 5 6 7]

'''

a = b  # a라는 변수에 b 값을 넣어라
a == b # a 와 b 가 같다.

'''


# x_test = x[7:]
# y_test = y[7:] 
# x_test = x[7:10]
# y_test = y[7:10] 
x_test = x[-3:]
y_test = y[-3:10]

# x_test = x[7:10:1] #[ 8  9 10]
# y_test = y[7:10:1] #[ 8  9 10]

print(x_train)
print(y_train)
print(x_test)
print(y_test)


#모델구성
model = Sequential()
model.add(Dense(1, input_dim = 1))
model.add(Dense(5))
model.add(Dense(1))


#컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train,y_train, epochs=1000, batch_size=1)

#평가, 예측
loss = model.evaluate(x_test,y_test)
result = model.predict([110000, 7])

print("로스 : ", loss)
print("예측 값: ", result)