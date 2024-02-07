from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
import numpy as np
import keras
import random as rn

print(tf.__version__) #2.9.0
print(keras.__version__) #2.9.0
print(np.__version__) #1.26.3
rn.seed(333)
tf.random.set_seed(123) # 텐서 2.9 먹힘. 2.15 안먹힘
np.random.seed(321)
#가중치의 초기값이 고정됨. ==> 그다음 연산이 다 동일함.
#가중치 값은 고정시키면 안된다는 학계의 정설. => 버전이 올라가며 변함
#result [2.5575156]

x = np.array([1,2,3])
y = np.array([1,2,3])

#2. 모델
model = Sequential()
model.add(Dense(5, 
                # kernel_initializer='zeros', 
                input_dim =1 ))
model.add(Dense(5))
model.add(Dense(1))

#3.컴파일, 훈련
model.compile(loss = "mse", optimizer="adam")
model.fit(x,y, epochs=100, batch_size=100)

#4.평가 예측
loss = model.evaluate(x,y)
results = model.predict([4], verbose=0)
print("loss :", loss)
print("result : ", results)