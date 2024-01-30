# best result
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, LSTM, GRU
import time
time_steps = 20
def split_xy(dataFrame, size):
    split_start_time = time.time()
    xs = []
    ys = [] 
    for i in range(len(dataFrame) - size ):
        x = dataFrame[i : (i + size)]
        y = dataFrame[i : (i + size)]['T (degC)']
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

x, y = split_xy(jena_csv, time_steps)  # spliting time :  5.13
print(x.shape, y.shape)

# 2. 모델 구성
model = Sequential()
model.add(LSTM(16, input_shape=(time_steps,14)))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(1))

# 3.컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x,y, epochs=100, batch_size=1000)

# 4. 평가, 예측
loss = model.evaluate(x,y)
print("loss : ", loss)
print("predict : \n" , model.predict(x))
