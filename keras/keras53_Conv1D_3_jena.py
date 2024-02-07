#5일분(720행)을 훈련시켜서 하루(144행)뒤 예측
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, LSTM, GRU, Bidirectional, Conv1D, Flatten
import time
from keras.callbacks import EarlyStopping

time_steps = 5 * 144
predict_standard_time = 144
def split_xy(dataFrame, cutting_size, y_behind_size,  y_column):
    split_start_time = time.time()
    xs = []
    ys = [] 
    for i in range(len(dataFrame) - cutting_size - y_behind_size):
        x = dataFrame[i : (i + cutting_size)]
        y = dataFrame[i + cutting_size + y_behind_size : (i + cutting_size + y_behind_size + 1) ][y_column]
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

x, y = split_xy(jena_csv, time_steps, predict_standard_time,'T (degC)')  # spliting time :  5.13
print(x.shape, y.shape) #(419687, 720, 14) (419687, 1)

# np.save(file='C:/_data/_save_npy/jena/jena_splited_x.npy', arr= x)
# np.save(file='C:/_data/_save_npy/jena/jena_splited_y.npy', arr= y)

# x = np.load(file='C:/_data/_save_npy/jena/jena_splited_x.npy')
# y = np.load(file='C:/_data/_save_npy/jena/jena_splited_y.npy')

#scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x = scaler.fit_transform(x.reshape(-1, x.shape[-1])).reshape(x.shape)

# 2. 모델 구성
model = Sequential()
model.add(Conv1D(64, kernel_size=(2,), input_shape=(time_steps,14)))
model.add(Conv1D(32, kernel_size=(2,), input_shape=(time_steps,14)))
model.add(Flatten())
model.add(Dense(32,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(1))

# 3.컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x,y, epochs=2000, batch_size=200, callbacks=[
    EarlyStopping(monitor='loss', mode = 'min', patience=100,restore_best_weights=True)
])

# 4. 평가, 예측
loss = model.evaluate(x,y)
print("loss : ", loss)
print("predict : \n" , model.predict(x))

