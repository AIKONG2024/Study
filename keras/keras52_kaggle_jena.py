# best result
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, LSTM, GRU, Bidirectional
import time
from keras.callbacks import EarlyStopping

time_steps = 1
def split_xy(dataFrame, size, y_column):
    split_start_time = time.time()
    xs = []
    ys = [] 
    for i in range(len(dataFrame) - size ):
        x = dataFrame[i : (i + size)]
        y = dataFrame[i : (i + size)][y_column]
        xs.append(x)
        ys.append(y)
    split_end_time = time.time()
    print("spliting time : ", np.round(split_end_time - split_start_time, 2),  "sec")
    return (np.array(xs), np.array(ys))

# 1. 데이터
path = "C:\_data\kaggle\jena\\"
jena_csv = pd.read_csv(path + "jena_climate_2009_2016.csv", index_col="Date Time")
print(jena_csv.shape)  # (420551, 14)

#확인
# print(dataFrame.head(1))

#결측치 확인
# print(np.unique(jena_csv.isna().sum()))

x, y = split_xy(jena_csv, time_steps, 'T (degC)')  # spliting time :  5.13
print(x.shape, y.shape)

#scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x = scaler.fit_transform(x)

# 2. 모델 구성
model = Sequential()
model.add(LSTM(64, input_shape=(time_steps,14)))
model.add(Dense(64,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(1))

# 3.컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x,y, epochs=2000, batch_size=5000, validation_split=0.1, callbacks=[
    EarlyStopping(monitor='val_loss', mode = 'min', patience=100,restore_best_weights=True)
])

# 4. 평가, 예측
loss = model.evaluate(x,y)
print("loss : ", loss)
print("predict : \n" , model.predict(x))

'''
loss :  0.004464844241738319
13143/13143 [==============================] - 10s 747us/step
predict : 
 [[-8.010506 ]
 [-8.392689 ]
 [-8.482526 ]
 ...
 [-3.3230095]
 [-3.1502385]
 [-4.2950053]]

'''