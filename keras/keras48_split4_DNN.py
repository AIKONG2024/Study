#DNN 으로 구성
#48_2 카피
#(N,4,1) -> (N,2,2) 로 변경해서 LSTM 구성
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
# x = x.reshape(-1, 4, 1)
print(x, y)
'''
 [[96 97]
  [98 99]]] [  5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20  21  22
  23  24  25  26  27  28  29  30  31  32  33  34  35  36  37  38  39  40
  41  42  43  44  45  46  47  48  49  50  51  52  53  54  55  56  57  58
  59  60  61  62  63  64  65  66  67  68  69  70  71  72  73  74  75  76
  77  78  79  80  81  82  83  84  85  86  87  88  89  90  91  92  93  94
  95  96  97  98  99 100]
'''
print(x.shape, y.shape)# (96, 4, 1) (96,)

x_predict = split_x(x_predict, size-1)
# x_predict = x_predict.reshape(-1,4,1)
# print(x_predict.shape) #(7, 4, 1)
print(x_predict)

#모델 구성및 평가 예측.
model = Sequential()
model.add(Dense(128, input_shape = (4,), activation='relu'))
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
 [[100.000046]
 [101.00003 ]
 [102.00003 ]
 [103.00002 ]
 [104.000015]
 [105.00002 ]
 [106.00002 ]]
'''