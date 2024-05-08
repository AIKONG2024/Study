import numpy as np
from keras.datasets import mnist
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Conv2D,Flatten, MaxPooling2D, GlobalAvgPool2D
from sklearn.preprocessing import OneHotEncoder

# 1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train / .255
x_test = x_test / .255
one_hot = OneHotEncoder()
y_train = one_hot.fit_transform(y_train.reshape(-1, 1)).toarray()
y_test = one_hot.transform(y_test.reshape(-1, 1)).toarray()

#2. 모델
model = Sequential()
model.add(Conv2D(8, (2,2), strides=2, padding= 'same', input_shape = (28, 28, 1))) 
# model.add(MaxPooling2D()) 
# model.add(Conv2D(filters=7,kernel_size=(2,2))) 
model.add(Conv2D(100,(2,2)))
model.add(GlobalAvgPool2D())
# model.add(Flatten())
model.add(Dense(50))
model.add(Dense(10, activation='softmax'))

model.summary()


#컴파일, 훈련
model.compile(loss= 'categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, batch_size=3000, verbose= 1, epochs= 100, validation_split=0.2 )

#4.평가, 예측
results = model.evaluate(x_test, y_test)
print('loss = ', results[0])
print('acc = ', results[1])

y_test_armg =  np.argmax(y_test, axis=1)
predict = np.argmax(model.predict(x_test),axis=1)
print(predict)