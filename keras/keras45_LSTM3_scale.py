import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.callbacks import EarlyStopping

#80 예측
#1. 데이터
x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
           [5,6,7],[6,7,8],[7,8,9],[8,9,10],
           [9,10,11],[10,11,12],
           [20,30,40],[30,40,50],[40,50,60]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
predict = np.array([50,60,70])

print(x.shape, y.shape)#(13, 3)(3,)
x = x.reshape(13,3,1)

#2. 모델구성
model = Sequential()
model.add(LSTM(70, input_shape = (3,1),))
model.add(Dense(64))
model.add(Dense(128))
model.add(Dense(256,activation='relu'))
model.add(Dense(512))
model.add(Dense(1))

#3. 모델 훈련, 컴파일
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
hist = model.fit(x,y,epochs=10000,batch_size=100, callbacks=[
    EarlyStopping(monitor='loss', mode='min', patience=500, restore_best_weights=True)
])

# 평가, 예측
loss = model.evaluate(x,y)
y_predict = model.predict(predict.reshape(1,3,1))

print("mse : ", loss[0])
print("mae : ", loss[1])
print("predict :", y_predict)


import matplotlib.pyplot as plt
plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], c ='red', label = 'loss', marker = '.')
plt.xlabel = 'epoch'
plt.ylabel = 'loss'
plt.grid()
plt.show()



'''
predict : [[78.74713]]
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.callbacks import EarlyStopping

#80 예측
#1. 데이터
x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
           [5,6,7],[6,7,8],[7,8,9],[8,9,10],
           [9,10,11],[10,11,12],
           [20,30,40],[30,40,50],[40,50,60]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
predict = np.array([50,60,70])

print(x.shape, y.shape)#(13, 3)(3,)
x = x.reshape(13,3,1)

#2. 모델구성
# model = Sequential()
# model.add(LSTM(140, input_shape = (3,1)))
# model.add(Dense(64,activation='relu'))
# model.add(Dense(200,activation='relu'))
# model.add(Dense(300,activation='relu'))
# model.add(Dense(400,activation='relu'))
# model.add(Dense(1))


model = Sequential()
model.add(LSTM(140, input_shape = (3,1)))
model.add(Dense(64,activation='relu'))
model.add(Dense(200,activation='relu'))
model.add(Dense(300,activation='relu'))
model.add(Dense(400,activation='relu'))
model.add(Dense(1))

#3. 모델 훈련, 컴파일
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
hist = model.fit(x,y,epochs=10000,batch_size=100, callbacks=[
    EarlyStopping(monitor='loss', mode='min', patience=300, restore_best_weights=True)
])

# 평가, 예측
loss = model.evaluate(x,y)
y_predict = model.predict(predict.reshape(1,3,1))

print("mse : ", loss[0])
print("mae : ", loss[1])
print("predict :", y_predict)
#predict : [[78.74713]]

import matplotlib.pyplot as plt
plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], c ='red', label = 'loss', marker = '.')
plt.xlabel = 'epoch'
plt.ylabel = 'loss'
plt.grid()
plt.show()
'''