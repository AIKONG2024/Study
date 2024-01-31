from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPooling2D, LSTM, Conv1D
import numpy as np
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score

#1. 데이터
#이미지 불러오기
npy_path = 'C:/_data/_save_npy/horse_human/'

#==================================================================================================

#binary
x_train = np.load(file= npy_path + 'keras39_07_save_x_train_horse_b_human.npy')
y_train = np.load(file= npy_path + 'keras39_07_save_y_train_horse_b_human.npy')

print(x_train.shape, y_train.shape) #(1027, 300, 300, 3) (1027,)
print(y_train)
# unique, count = np.unique(y_train, return_counts=True)
# print(unique, count)#[0. 1.] [500 527]


x_train, x_test, y_train, y_test =  train_test_split(x_train, y_train, train_size=0.8, shuffle=True, random_state=3123)

x_train = x_train.reshape(-1,900,300)
x_test = x_test.reshape(-1,900,300)

#2. 모델 구성
model = Sequential()
model.add(Conv1D(16, 2, input_shape = (900,300) , activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.summary()

# #컴파일 ,훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=4, batch_size=100, validation_split=0.2, callbacks=[
    EarlyStopping(monitor='val_loss', mode='min', patience=100, restore_best_weights=True)
])

# 평가 예측
arg_test_y = np.round(y_test)
predict = np.round(model.predict(x_test))
loss  =  model.evaluate(x_test, y_test)

acc_score = accuracy_score(arg_test_y, predict)

print(' acc : ', acc_score)
print(predict)
'''
acc : 1.0
=============RNN 적용
 acc :  1.0
[[0.]
 [0.]]
 ==============Conv1D
  acc :  0.8349514563106796
'''
