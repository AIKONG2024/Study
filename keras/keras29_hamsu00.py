import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense , Input

# 1. 데이터(잘못 준 데이터) -> 행열 원위치
x = np.array(
    [
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3] # 짜증나는 데이터지만, 로스가 떨어지는 것을 확인하면서 가중치를 구할 수 있음.
    ]  #컬럼 데이터들을 대괄호로 묶어 줘야함.
)
y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

print(x.shape)  # (2,10)
print(y.shape)  # (10,)

x = x.swapaxes(0,1) 
print(x.shape)  # (10, 2)
print(x)
# 2. 모델 구성(순차적)
model = Sequential()
model.add(Dense(10, input_dim = 2)) #input 데이터는 2차원. 
model.add(Dense(9))
model.add(Dense(8))
model.add(Dense(7))
model.add(Dense(1))
model.summary()
'''
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 dense (Dense)               (None, 10)                30

 dense_1 (Dense)             (None, 9)                 99

 dense_2 (Dense)             (None, 8)                 80

 dense_3 (Dense)             (None, 7)                 63

 dense_4 (Dense)             (None, 1)                 8

=================================================================
Total params: 280 (1.09 KB)
Trainable params: 280 (1.09 KB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________
'''
# 2. 모델 구성(함수형)
input1 = Input(shape=(2,)) # == input_shape
dense1 = Dense(10)(input1) # input1 인풋 레이어 연결
dense2 = Dense(9)(dense1) #dense1 연결
dense3 = Dense(8)(dense2) #dense2 연결
dense4 = Dense(7)(dense3) #dense3 연결
output1 = Dense(1)(dense4) #dense4 연결
model = Model(inputs = input1, outputs = output1)
'''

Model: "model"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 input_1 (InputLayer)        [(None, 2)]               0

 dense_5 (Dense)             (None, 10)                30

 dense_6 (Dense)             (None, 9)                 99

 dense_7 (Dense)             (None, 8)                 80

 dense_8 (Dense)             (None, 7)                 63

 dense_9 (Dense)             (None, 1)                 8

=================================================================
Total params: 280 (1.09 KB)
Trainable params: 280 (1.09 KB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________
'''

model.summary()

# #3.컴파일, 훈련
# model.compile(loss='mse', optimizer='adam')
# model.fit(x,y, epochs=3000, batch_size= 1)
# #4.평가, 예측
# loss = model.evaluate(x,y)
# results = model.predict([[10, 1.3]]) 

# print("[10, 1.3]의 예측값 : ", results) 