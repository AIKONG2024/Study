# from tensorflow.keras.models import Sequential
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import tensorflow as tf
import keras

print("tf 버전 : ", tf.__version__)
print("keras 버전 : ", keras.__version__)

# 1. 데이터
x = np.array([1, 2, 3, 4, 5, 6])
y = np.array([1, 2, 3, 5, 4, 6])

# 레이어 5개 정도 잡고, batch size 수정 해서, epoch 0.31999정도

# 2. 모델구성
model = Sequential()
model.add(Dense(1, input_dim=1))  # 통상 레이어에 노드 1개를 넣지 않는다.
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss="mse", optimizer="adam")
model.fit(
    x, y, epochs=200, batch_size=6
)  # batch_size: 데이터를 1개씩 잘라서 훈련하겠다. 일괄처리할 수 있는 사이즈. epochs * batch_size 만큼 훈련함.

# 4. 평가, 예측
loss = model.evaluate(x, y)
result = model.predict([7])

print("로스 : ", loss)
print("7의 예측값 : ", result)

# layer 3, dense 12000, epoch = 100, batch_size = default,
# 로스 :  0.32381489872932434
# 7의 예측값 :  [[6.795264]]

# layer 3, dense 12000, epoch = 100, batch_size = default,
# 로스 :  0.3238097131252289
# 7의 예측값 :  [[6.800002]]

# layer 3, dense 12000, epoch = 10000, batch_size = default,
# 로스 :  0.3238079249858856
# 7의 예측값 :  [[6.799952]]
