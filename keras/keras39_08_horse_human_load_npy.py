from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPooling2D
import numpy as np
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score

#1. 데이터
#이미지 불러오기
npy_path = 'C:\_data\_save_npy\horse_human/'
x_train = np.load(file= npy_path + 'keras39_07_save_x_train_horse_human.npy')
y_train = np.load(file= npy_path + 'keras39_07_save_y_train_horse_human.npy')

print(x_train.shape, y_train.shape) #(1027, 300, 300, 3) (1027, 2)

unique, count = np.unique(y_train, return_counts=True)
print(unique, count)#[0. 1.] [1027 1027]

x_train, x_test, y_train, y_test =  train_test_split(x_train, y_train, train_size=0.8, shuffle=True, random_state=777)

#2. 모델 구성
model = Sequential()
model.add(Conv2D(32, kernel_size=(2,2), input_shape = (x_train.shape[1], x_train.shape[2], x_train.shape[3]) , activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
model.add(Dropout(0.2))    

model.add(Conv2D(64, (2,2), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
model.add(Dropout(0.3))
    
model.add(Conv2D(126, (2,2), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

#컴파일 ,훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=1000, batch_size=100, validation_split=0.2, callbacks=[
    EarlyStopping(monitor='val_loss', mode='min', patience=100, restore_best_weights=True)
])

# 평가 예측
arg_test_y = np.argmax(y_test, axis=1)
predict = np.argmax(model.predict(x_test), axis=1)
loss  =  model.evaluate(x_test, y_test)

acc_score = accuracy_score(arg_test_y, predict)

print(' acc : ', acc_score)
print(predict)

'''
 acc :  1.0
[1 0 1 0 0 1 0 1 1 1 1 1 0 1 1 0 0 1 0 0 1 1 1 0 0 0 1 0 0 1 1 0 1 1 0 0 1
 1 0 1 0 0 0 0 1 1 0 1 0 0 1 1 1 0 0 1 0 1 0 1 0 1 0 1 0 0 0 0 1 0 0 1 1 1
 0 1 0 1 0 0 1 1 0 0 1 1 0 1 1 0 1 1 0 1 1 1 1 0 0 0 0 1 1 1 1 1 0 1 1 1 0
 1 1 0 1 0 1 1 0 0 1 1 1 0 1 0 0 0 1 1 1 1 0 0 1 0 0 1 0 0 0 0 0 0 1 0 1 1
 1 0 1 0 0 1 1 1 1 0 0 1 0 0 0 1 0 0 0 1 0 1 1 1 0 0 0 1 1 1 0 0 0 0 0 0 1
 0 1 0 0 1 0 1 1 0 0 0 0 1 1 0 0 1 0 1 1 1]
'''