from keras.models import Sequential
from keras.layers import Dense
import numpy as np

# 1.데이터
x = np.array([1,2,3,4,5,6])
y = np.array([1,2,3,5,4,6])

# 요 데이터를 훈련해서 최소의 loss를 만들기

# 2.모델구성
#### [실습] 100ephch에 01_1번과 같은 결과를 빼기
model = Sequential()
model.add(Dense(1, input_dim = 1))

# 3.컴파일,훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100)

# 평가,예측 (7)
loss = model.evaluate(x,y)

print("로스: ", loss) #많이 돌려도 핑퐁치는 구간이 발생할 수 있다.

result = model.predict([1,2,3,4,5,6,7])
print("예측: ", result)

#0.33이하 로스값 만들기
# 로스:  0.3238094747066498
# 1/1 [==============================] - 0s 47ms/step
# 예측:  [[1.1428571]
#  [2.0857143]
#  [3.0285714]
#  [3.9714286]
#  [4.9142857]
#  [5.857143 ]
#  [6.8      ]]