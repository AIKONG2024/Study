#acc = 0.77 이상

from keras.datasets import cifar10
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from sklearn.preprocessing import OneHotEncoder
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

#one hot encoder
onehot = OneHotEncoder(sparse=False)
y_train = onehot.fit_transform(y_train)
y_test = onehot.fit_transform(y_test)

#2. 모델구성
model = Sequential()
model.add(Conv2D(filters=4, kernel_size=(2,2), input_shape = (32,32,3)))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(Conv2D(32, (4,4), activation='relu'))
model.add(Conv2D(64, (5,5), activation='relu'))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(10, activation='relu'))

#3. 컴파일 훈련
model.compile(loss= 'categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=10, batch_size=5000, verbose=2, validation_split=0.2)

#4. 평가, 예측
results = model.evaluate(x_test, y_test)
predict = np.argmax( model.predict(x_test),axis=1)
acc_score =  accuracy_score(np.argmax(y_test, axis=1), predict)
print('loss = ', results[0])
print('acc = ', results[1])
print(predict)

