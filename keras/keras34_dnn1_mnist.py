import numpy as np
from keras.datasets import mnist
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Conv2D,Flatten, MaxPooling2D
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import OneHotEncoder

# 1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2] * x_train.shape[3])
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2] * x_test.shape[3])

one_hot = OneHotEncoder()
y_train = one_hot.fit_transform(y_train.reshape(-1, 1)).toarray()
y_test = one_hot.transform(y_test.reshape(-1, 1)).toarray()

#2. 모델
model = Sequential()
# model.add(Flatten())
model.add(Dense(32, input_shape = (x_train.shape[1],)))
model.add(Dense(64,activation='relu'))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(64,activation='relu'))
# model.add(Dense(32,activation='relu'))
# model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.summary()

es = EarlyStopping(monitor='val_loss', mode='min', patience=30, restore_best_weights=True)

#컴파일, 훈련
model.compile(loss= 'categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, batch_size=1000, verbose= 1, epochs= 1000, validation_split=0.2, callbacks=[es] )

#4.평가, 예측
results = model.evaluate(x_test, y_test)
print('loss = ', results[0])
print('acc = ', results[1])

y_test_armg =  np.argmax(y_test, axis=1)
predict = np.argmax(model.predict(x_test),axis=1)
print(predict)

'''
loss =  0.18924179673194885
acc =  0.9858999848365784

dnn 결과 =====================================
loss =  1.2765123844146729
acc =  0.9696000218391418
313/313 [==============================] - 0s 507us/step
[7 2 1 ... 4 5 6]
'''