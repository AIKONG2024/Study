import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.callbacks import EarlyStopping

a = np.array(range(1, 101))
x_predict = np.array(range(96,106))  # 

size = 5  #x데이터는 4개, y 데이터는 1개
#4개씩  timesteps
def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1): 
        subset = dataset[i : (i + size)] 
        aaa.append(subset) 
    return np.array(aaa)


xy_splited = split_x(a, size)
print(x_predict.shape) #(10,)

x = xy_splited[:, :-1]
y = xy_splited[:, -1]
x = x.reshape(-1, 4, 1)
print(x, y)
print(x.shape, y.shape) #(96, 4) (96,)

x_predict = split_x(x_predict, size-1)
print(x_predict.shape) #(7, 4)
print(x_predict)
'''
spilted x_predict
[[ 96  97  98  99]
 [ 97  98  99 100]
 [ 98  99 100 101]
 [ 99 100 101 102]
 [100 101 102 103]
 [101 102 103 104]
 [102 103 104 105]]
'''

#모델 구성및 평가 예측.
model = Sequential()
model.add(LSTM(128, input_shape = (4,1), activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(3, activation='relu'))
model.add(Dense(1))

model.compile(loss = 'mse', optimizer='adam')
model.fit(x, y, epochs=10000, callbacks=[
    EarlyStopping(monitor='loss', mode='min', patience=1000, restore_best_weights=True)
])

loss = model.evaluate(x)
print('loss: ', loss)
predict = model.predict(x_predict)
print('predict : \n', predict) #100, 101, 102, 103, 104, 105, 106

'''
predict : 
loss:  0.0 
[[100.00262 ]
 [101.00351 ]
 [102.004456]
 [103.00549 ]
 [104.0066  ]
 [105.00776 ]
 [106.00899 ]]
'''