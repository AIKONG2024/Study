import tensorflow as tf # tensorflow를 땡겨오고, tf라고 줄여서 쓴다.
print(tf.__version__)   # 2.15.0
from keras.models import Sequential #순차적 모델을 땡겨옴
from keras.layers import Dense #Dense(아직 설명x)를 땡겨옴
import numpy as np

#1. 데이터
x = np.array([1,2,3]) #numpy : 속도, 데이터작업 용이
y = np.array([1,2,3])

# 2.모델구성
model = Sequential() #텐서플로우의 순차적 모델 생성
model.add(Dense(1, input_dim = 1)) #1개의 데이터 덩어리(Dense)모델을 넣어줌. Dense(output(y), input(x))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam') #두 값의 loss값을 mse(제곱하는 방식으로 양수를 만드는)라는 수식으로 씀
# optimizer='adam'은 건들지 않음. 그냥 씀.
model.fit(x,y, epochs = 8600) #fit() 훈련시키기. epochs: 훈련을 너무 많이 시키면 과적합 됨.-> 터짐. 훈련양을 조절해야함. 10번으로 조절, 이 과정으로 최적의 웨이트가 생성

# 4. 평가, 예측
loss = model.evaluate(x,y) #model에는 최적의 웨이트가 생성되어 있음.
print("로스 : ", loss)
result = model.predict([4])
print("4의 예측값 : ", result)


# 2200번정도 돌려야 좋았음

