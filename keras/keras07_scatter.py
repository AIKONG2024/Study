import numpy as np
from keras.models import Sequential
from keras.layers import Dense

from sklearn.model_selection import train_test_split

# 1. 데이터
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([1, 2, 3, 4, 6, 5, 7, 8, 9, 10])

#데이터 전처리
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size= 0.7,
    test_size=0.3,
    shuffle=True,
    random_state=1500
)

print(x_train)
print(y_train)
print(x_test)
print(y_test)

#모델구성
model = Sequential()
model.add(Dense(1, input_dim = 1))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

#컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=500 , batch_size=1)

#평가 , 예측
loss = model.evaluate(x_test, y_test)
result = model.predict(x) #10개가 들어감.

print("loss 값: ", loss)
print("예측 값: ", result)

import matplotlib.pyplot as plt

plt.scatter(x,y) #점 찍기
plt.plot(x, result, color = 'red') #선을 긋기
plt.show() #그래프 보이기
