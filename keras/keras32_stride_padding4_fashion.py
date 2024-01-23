#acc 0.95
from keras.datasets import fashion_mnist
import matplotlib.pylab as plt
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, MaxPooling2D, Flatten
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score
import time

#1.데이터
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

#scaling
x_train = x_train/255
x_test = x_test/255

# plt.imshow(x_train[0])
# plt.show()
print(x_train.shape)#(60000, 28, 28)
print(y_train.shape)#(60000,)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
ohe = OneHotEncoder(sparse=False)
y_train = ohe.fit_transform(y_train.reshape(-1, 1))
y_test = ohe.transform(y_test.reshape(-1, 1))

print(y_train.shape)
print(y_test.shape)

#2.모델구성
model = Sequential()
model.add(Conv2D(64, (2,2), input_shape = (28,28,1), strides=2, padding='same', activation='relu'))
model.add(Dropout(0.15))

model.add(Conv2D(128, (2,2), strides=2,padding='same', activation='relu'))
model.add(Dropout(0.3))

model.add(Conv2D(256, (2,2), activation='relu'))
model.add(Dropout(0.5))

model.add(Flatten())

#certificate
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

#3.모델 컴파일, 훈련
es = EarlyStopping(monitor='val_loss', mode='min', patience=30, restore_best_weights=True)
mcp = ModelCheckpoint(filepath="C:\_data\_save\MCP\cnn7_fashion\cnn_7_fashion_{epoch:04d}_{val_loss:4f}.h5", monitor='val_loss',
                      save_best_only=True, mode= 'min', initial_value_threshold=0.26)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
startTime = time.time()
model.fit(x_train, y_train, epochs=300, batch_size= 400, validation_split=0.2, callbacks=[es, mcp])
endTime = time.time()

#4.모델 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss :', loss[0])
print('acc :', loss[1])

predict = ohe.inverse_transform(model.predict(x_test))
print(predict)

acc_score = accuracy_score(ohe.inverse_transform(y_test), predict)
print('acc_score :', acc_score)

print("time :", round(endTime - startTime, 2), '초')


'''
loss : 0.24434790015220642
acc : 0.9194999933242798
313/313 [==============================] - 1s 2ms/step
[[9]
 [2]
 [1]
 ...
 [8]
 [1]
 [5]]
acc_score : 0.9195
time : 266.45 초

=========stride, padding 적용 후=============
[[9]
 [2]
 [1]
 ...
 [8]
 [1]
 [5]]
acc_score : 0.9242
time : 57.98 초
'''