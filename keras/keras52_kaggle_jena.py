# best result
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, LSTM, GRU
import time

time_steps = 20
def split_x(dataFrame, size):
    split_start_time = time.time()
    aaa = []
    for i in range(len(dataFrame) - size + 1):
        subset = dataFrame[i : (i + size)]
        aaa.append(subset)
    split_end_time = time.time()
    print("time : ", np.round(split_end_time - split_start_time, 2),  "sec")
    return np.array(aaa)

# 1. 데이터
path = "C:\_data\kaggle\jena\\"
jena_csv = pd.read_csv(path + "jena_climate_2009_2016.csv", index_col="Date Time")
print(jena_csv.shape)  # (420551, 14)

#T (degC) 맨뒤로 보냄
temp = jena_csv['T (degC)']
dataFrame = jena_csv.drop(columns=['T (degC)'])
dataFrame['T (degC)'] = temp
#확인
# print(dataFrame.head(1))

#결측치 확인
# print(np.unique(jena_csv.isna().sum()))

splited_csv = split_x(jena_csv, time_steps)  # spliting time :  5.13
print(splited_csv.shape)  # (420551, 14)

x = splited_csv[:,:-1]
y = splited_csv[:,-1]

print(x.shape, y.shape)  # (420532, 19, 14) (420532, 14)

# 2. 모델 구성
model = Sequential()
model.add(SimpleRNN(16, input_shape=(19,14)))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(1))

# 3.컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x,y, epochs=100, batch_size=500)

# 4. 평가, 예측
model.evaluate(x,y)
print("predict : \n" , model.predict(x))
