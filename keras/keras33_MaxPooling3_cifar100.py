#acc 0.4 이상

from keras.datasets import cifar100
from keras.models import Sequential
from keras.layers import Dense, Conv2D , Dropout, Flatten, MaxPooling2D
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import OneHotEncoder
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

#.2 모델생성

#Conv Layer
model = Sequential(Conv2D(64, (2,2), input_shape = (32,32,3), activation='relu'))
model.add(MaxPooling2D(strides=(2,2)))
model.add(Dropout(0.1))

model.add(Conv2D(128, (2,2), activation='relu'))
model.add(MaxPooling2D(strides=(2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(256, (2,2), activation='relu'))
model.add(MaxPooling2D(strides=(2,2)))
model.add(Dropout(0.3))

model.add(Conv2D(512, (2,2), activation='relu'))
model.add(MaxPooling2D(strides=(2,2)))
model.add(Dropout(0.3))
#Flatten
model.add(Flatten())

#Certification
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(100, activation='softmax'))

es = EarlyStopping(monitor='val_loss', mode='min', patience=50, restore_best_weights=True)
mcp = ModelCheckpoint(filepath="C:\_data\_save\MCP\cifar100\cifar100_{epoch:04d}_{val_loss:4f}.h5", monitor='val_loss', 
                      mode='min', save_best_only=True, initial_value_threshold=2.4 )

#3.모델 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
import time
startTime = time.time()
model.fit(x_train, y_train, epochs=200, batch_size=300, validation_split=0.2, callbacks=[es, mcp])
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


MaxPooling 사용 
loss:  1.8620822429656982
acc:  0.5063999891281128
313/313 [==============================] - 0s 850us/step
[[95]
 [80]
 [55]
 ...
 [51]
 [42]
 [45]]
acc_score :  0.5064
elapsed time :  231.49 sec
'''