# [실습]
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# 1.데이터
x = np.array(
    [
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3],
        [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    ]
)

y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
print(x)
print(x.shape, y.shape)  # shape : 구조를 보여줌 (2, 10) (10,)
x = x.T
print(x.shape)  # (10, 2)

#모델 구성
model = Sequential()
model.add(Dense(1, input_dim = 3))
# model.add(Dense(1))
# model.add(Dense(1))

#컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x,y, epochs= 3000, batch_size=1)

#평가
loss = model.evaluate(x,y)
result = model.predict([[10,1.3,0]])


print("로스 : ", loss)
print("예측값 : ", result)

# deep = 1(3-1), batch_size = 1, epochs = 3000
# 로스 :  5.928768587182276e-11
# 예측값 :  [[9.999992]]
# 단층 레이어에서 성능이 가장 잘나왔음 ==> 아마 데이터가 적어서 그런 것 같음.(아직은 딥러닝 수준이 아님 단지 예제)