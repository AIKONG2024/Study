#acc = 0.77 이상

from keras.datasets import cifar10
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import OneHotEncoder, StandardScaler
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

#2. 모델구성

model = Sequential()
# Block 1
model.add(Conv2D(150, (2, 2), activation='relu', input_shape=(32, 32, 3)))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))
model.add(Dropout(0.3))

# Block 2
model.add(Conv2D(150, (2, 2), activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))
model.add(Dropout(0.3))

# Block 3
model.add(Conv2D(150, (2, 2), activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))
model.add(Dropout(0.3))

# Classification block
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))  # Use softmax activation for 10 classes

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


'''
loss =  0.6011049151420593
acc =  0.8007000088691711
acc_score =  0.8007
=========stride, padding 적용 후=============

'''