#acc 0.4 이상

from keras.datasets import cifar100
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D , Dropout, Flatten, MaxPooling2D, Input
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
print(x_train.shape, y_train.shape) #(50000, 32, 32, 3) (50000, 1)
unique, count = np.unique(y_test, return_counts= True)
print(unique, count) 

#scailing
x_train = x_train/255
x_test = x_test/255

#encoding
ohe = OneHotEncoder(sparse=False)
y_train = ohe.fit_transform(y_train)
y_test = ohe.fit_transform(y_test)

print(y_train.shape)

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

#.2 모델생성

# #Conv Layer
# model = Sequential(Conv2D(64, (2,2), input_shape = (32,32,3), activation='relu'))
# # model.add(MaxPooling2D((2,2), strides=(2,2)))
# model.add(Dropout(0.1))

# model.add(Conv2D(128, (2,2), activation='relu'))
# # model.add(MaxPooling2D((2,2), strides=(2,2)))
# model.add(Dropout(0.2))

# model.add(Conv2D(256, (2,2), activation='relu'))
# # model.add(MaxPooling2D((2,2), strides=(2,2)))
# model.add(Dropout(0.3))

# model.add(Conv2D(512, (2,2), activation='relu'))
# # model.add(MaxPooling2D((2,2), strides=(2,2)))
# model.add(Dropout(0.3))
# #Flatten
# model.add(Flatten())

# #Certification
# model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(100, activation='softmax'))


#함수형
input_l = Input(shape=(32,32,3))
conv2d_l1 = Conv2D(128, (2, 2), activation='relu')(input_l)
dropo_l1 = Dropout(0.3)(conv2d_l1)

conv2d_l2 = Conv2D(256, (2, 2), activation='relu')(dropo_l1)
dropo_l2 = Dropout(0.3)(conv2d_l2)

conv2d_l3 = Conv2D(512, (2, 2), activation='relu')(dropo_l2)
dropo_l3 = Dropout(0.3)(conv2d_l3)

flat_l = Flatten()(dropo_l3)
d_11 = Dense(256, activation='relu')(flat_l)
drop_l4 = Dropout(0.5)(d_11)
output_l = Dense(100, activation='softmax')(drop_l4)
model = Model(inputs = input_l, outputs = output_l)



es = EarlyStopping(monitor='val_loss', mode='min', patience=50, restore_best_weights=True)
mcp = ModelCheckpoint(filepath="C:\_data\_save\MCP\cifar100\cifar100_{epoch:04d}_{val_loss:4f}.h5", monitor='val_loss', 
                      mode='min', save_best_only=True, initial_value_threshold=2.4 )

#3.모델 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
import time
startTime = time.time()
model.fit(x_train, y_train, epochs=500, batch_size=50, validation_split=0.2, callbacks=[es, mcp])
endTime = time.time()

#평가, 예측
loss = model.evaluate(x_test, y_test)

print('loss: ', loss[0])
print('acc: ', loss[1])

predict = ohe.inverse_transform(model.predict(x_test))
print(predict)

acc_score = accuracy_score(ohe.inverse_transform(y_test), predict)
print("acc_score : ", acc_score)
print("elapsed time : ", round(endTime - startTime,2), "sec")

'''
loss:  2.1419894695281982
acc:  0.44449999928474426
313/313 [==============================] - 0s 980us/step
[[12]
 [80]
 [ 4]
 ...
 [51]
 [42]
 [70]]
acc_score :  0.4445
elapsed time :  412.17 sec
'''

'''
loss:  2.2689476013183594
acc:  0.4246000051498413
313/313 [==============================] - 0s 893us/step
[[12]
 [80]
 [30]
 ...
 [96]
 [42]
 [45]]
acc_score :  0.4246

======스케일링 적용 후
loss:  2.8446741104125977
acc:  0.3084999918937683
313/313 [==============================] - 2s 5ms/step
[[30]
 [80]
 [72]
 ...
 [51]
 [26]
 [70]]
acc_score :  0.3085
'''