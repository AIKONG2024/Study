from keras.models import Sequential
from keras.layers import Dense
import numpy as np

# 1.데이터
x = np.array([1,2,3,4,5,6])
y = np.array([1,2,3,5,4,6])

#과제2)요 데이터를 훈련해서 최소의 loss를 만들기
# 0.33이하 로스값 만들기

# 2.모델구성
#### [실습] 100ephch에 01_1번과 같은 결과를 빼기
model = Sequential()
model.add(Dense(1, input_dim = 1))
model.add(Dense(10000))
model.add(Dense(1))

# 3.컴파일,훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100)

# 평가,예측 (7)
loss = model.evaluate(x,y)

print("로스: ", loss) #많이 돌려도 핑퐁치는 구간이 발생할 수 있다.

result = model.predict([1,2,3,4,5,6,7])
print("예측: ", result)

#결과====================================================================
# 로스:  0.32422947883605957
# 1/1 [==============================] - 0s 54ms/step
# 예측:  [[1.1071261]
#  [2.058506 ]
#  [3.009886 ]
#  [3.9612646]
#  [4.9126444]
#  [5.864025 ]
#  [6.815401 ]]
#=========================================================================