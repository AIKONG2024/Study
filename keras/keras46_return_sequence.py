import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
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
model.add(LSTM(16, input_shape = (3,1), return_sequences=True)) 
#timesteps 를 그대로 살려서 전달해서 LSTM 을 연달아 사용가능
# model.add(Dropout(0.2))
model.add(LSTM(10, return_sequences=True))
model.add(LSTM(10, return_sequences=True))
model.add(LSTM(10, return_sequences=True))
model.add(LSTM(10, return_sequences=True))
model.add(LSTM(10))
# model.add(LSTM(8, return_sequences=True))
# model.add(LSTM(4))
# model.add(Dense(64,activation='relu'))
# model.add(Dense(200,activation='relu'))
# model.add(Dense(300,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(1))

model.summary()

#3. 모델 훈련, 컴파일
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
hist = model.fit(x,y,epochs=10000,batch_size=100, callbacks=[
    EarlyStopping(monitor='loss', mode='min', patience=300, restore_best_weights=True)
])

# 평가, 예측
loss = model.evaluate(x,y)
x_predict = model.predict(predict.reshape(1,3,1))

print("mse : ", loss[0])
print("mae : ", loss[1])
print("predict :", x_predict)

# import matplotlib.pyplot as plt
# plt.figure(figsize=(9,6))
# plt.plot(hist.history['loss'], c ='red', label = 'loss', marker = '.')
# plt.xlabel = 'epoch'
# plt.ylabel = 'loss'
# plt.grid()
# plt.show()

'''

#mse :  3.92831739359778e-10
#predict : [[73.90749]]

mse :  7.322857982217101e-06
mae :  0.001997984480112791
predict : [[78.0333]]

mse :  1.516269492185529e-07
mae :  0.0002059569669654593
predict : [[77.21593]]

mse :  0.0002241086622234434
mae :  0.011969382874667645
predict : [[71.43323]]
'''