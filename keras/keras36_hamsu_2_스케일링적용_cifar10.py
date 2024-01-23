#acc = 0.77 이상

from keras.datasets import cifar10
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(x_train.shape, y_train.shape)#(50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)#(10000, 32, 32, 3) (10000, 1)
unique, count = np.unique(y_train, return_counts=True)
#[0 1 2 3 4 5 6 7 8 9] [5000 5000 5000 5000 5000 5000 5000 5000 5000 5000]
print(unique, count)
'''
Label	Description
0	airplane
1	automobile
2	bird
3	cat
4	deer
5	dog
6	frog
7	horse
8	ship
9	truck
'''
#(m,32,32,3)
print(x_train.shape[0])
print(x_train.shape[1])
print(x_train.shape[2])

#scaling
x_train = x_train / 255.0 
x_test = x_test /255.0 

#encoding
onehot = OneHotEncoder(sparse=False)
y_train = onehot.fit_transform(y_train)
y_test = onehot.fit_transform(y_test)

#스케일링 1-1 
# x_train 과 x_test 의 fit된 모양이 같아야하지만
# 이미지에서는 0~ 255이므로 255로 나눠도 괜찮음. == Minmax 
# x_train = x_train/255.
# x_test = x_test/255.#0~1로 정규화함

# #스케일링 1-2
# #Standard 
# x_train = (x_train - 127.5)/ 127.5
# x_test = (x_test - 127.5)/ 127.5 #-1~1 로 일반화함. 0을 기준으로 정규분포 모양을 만들기 위해서. 실질적인 standard scaler는 아님.


x_train = x_train.reshape(-1,32*32*3)
x_test = x_test.reshape(-1, 32*32*3)

#스케일링 2-1
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

#스케일링 2-2
# scaler = StandardScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.fit_transform(x_test)

#
x_train = x_train.reshape(-1,32,32,3)
x_test = x_test.reshape(-1,32,32,3)



#2. 모델구성

# model = Sequential()
# # Block 1
# model.add(Conv2D(150, (2, 2), activation='relu', input_shape=(32, 32, 3)))
# model.add(MaxPooling2D((2, 2), strides=(2, 2)))
# model.add(Dropout(0.3))

# # Block 2
# model.add(Conv2D(150, (2, 2), activation='relu'))
# model.add(MaxPooling2D((2, 2), strides=(2, 2)))
# model.add(Dropout(0.3))

# # Block 3
# model.add(Conv2D(150, (2, 2), activation='relu'))
# model.add(MaxPooling2D((2, 2), strides=(2, 2)))
# model.add(Dropout(0.3))

# # Classification block
# model.add(Flatten())
# model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(10, activation='softmax'))  # Use softmax activation for 10 classes

#함수형
input_l = Input(shape=(32,32,3))
conv2d_l1 = Conv2D(150, (2, 2), activation='relu')(input_l)
mp2d_l1= MaxPooling2D((2, 2), strides=(2, 2))(conv2d_l1)
dropo_l1 = Dropout(0.3)(mp2d_l1)

conv2d_l2 = Conv2D(150, (2, 2), activation='relu')(dropo_l1)
mp2d_l2 = MaxPooling2D((2, 2), strides=(2, 2))(conv2d_l2)
dropo_l2 = Dropout(0.3)(mp2d_l2)

conv2d_l3 = Conv2D(150, (2, 2), activation='relu')(dropo_l2)
mp2d_l3 = MaxPooling2D((2, 2), strides=(2, 2))(conv2d_l3)
dropo_l3 = Dropout(0.3)(mp2d_l3)

flat_l = Flatten()(dropo_l3)
d_11 = Dense(256, activation='relu')(flat_l)
drop_l4 = Dropout(0.5)(d_11)
output_l = Dense(10, activation='softmax')(drop_l4)
model = Model(inputs = input_l, outputs = output_l)

#EarlyStop
es = EarlyStopping(monitor='val_loss', mode='min', patience=30, restore_best_weights=True)

#3. 컴파일 훈련
model.compile(loss= 'categorical_crossentropy', optimizer='adam', metrics=['acc'])
import time
startTime = time.time()
history = model.fit(x_train, y_train, epochs=500, batch_size=32, verbose=2, validation_split=0.2, callbacks=[es])
endTime = time.time()

#모델저장
model.save("..\_data\_save\cifar10\keras31_cann5_save_model.h5") 

#4. 평가, 예측
results = model.evaluate(x_test, y_test)
predict = np.argmax( model.predict(x_test),axis=1)
acc_score =  accuracy_score(np.argmax(y_test, axis=1), predict)
print('loss = ', results[0])
print('acc = ', results[1])
print('acc_score = ', acc_score)
print(predict)

#시각화
import matplotlib.pyplot as plt
plt.figure(figsize=(9,6))
plt.plot(history.history['val_acc'], color = 'blue', label = 'val_acc', marker = '.')
plt.plot(history.history['val_loss'], color = 'red', label = 'val_loss', marker = '.')
plt.show()
