import numpy as np
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, Dropout, LSTM
from keras.callbacks import EarlyStopping

# 1. 데이터
datasets = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# timesteps :3
x = np.array(
    [
        [1, 2, 3],
        [2, 3, 4],
        [3, 4, 5],
        [4, 5, 6],
        [5, 6, 7],
        [6, 7, 8],
        [7, 8, 9],
    ]
)

y = np.array([4, 5, 6, 7, 8, 9, 10])

print(x.shape, y.shape) #(7, 3) (7,)
x = x.reshape(7, 3, 1)
print(x.shape, y.shape) #(7, 3, 1) (7,)

#2. 모델구성
model = Sequential()
# model.add(SimpleRNN(units=4, input_shape=(3, 1), activation='relu')) # timesteps, features
model.add(LSTM(units=10, input_shape=(3, 1), activation='relu')) # timesteps, features
model.add(Dense(7, activation='relu'))
model.add(Dense(1))

model.summary()
'''
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 lstm (LSTM)                 (None, 10)                480

 dense (Dense)               (None, 7)                 77

 dense_1 (Dense)             (None, 1)                 8

=================================================================
Total params: 565
Trainable params: 565
Non-trainable params: 0
_________________________________________________________________
parameters = 4 * ((n * m) + (n * n) + n)
'''



# #3. 컴파일, 훈련
# model.compile(loss='mse', optimizer='adam', metrics=["mae"])
# model.fit(x, y, epochs=2000, batch_size=4,validation_split=0.2, callbacks=[
#     EarlyStopping(monitor='val_loss', mode='min', patience=1000, restore_best_weights=True)
# ])

# # 4. 평가 예측
# result = model.evaluate(x,y)
# print('loss mse: ', result[0])
# print('loss mae: ', result[1])
# #(3,) -> (1,3,1)
# y_pred = np.array([8,9,10]).reshape(1,3,1)
# y_pred = model.predict(y_pred)
# print("predict : ", y_pred)
# #predict :  [[11.000721]]