from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense,Dropout,Conv2D, MaxPooling2D, Flatten
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import OneHotEncoder
from keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape, y_train.shape)#(50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)#(10000, 32, 32, 3) (10000, 1)

data_generator = ImageDataGenerator(
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    shear_range=20
)

augumet_size = 30000

randidx = np.random.randint(x_train.shape[0], size= augumet_size)

x_augumeted = x_train[randidx].copy()
y_augumeted = y_train[randidx].copy()

print(x_augumeted.shape)

x_augumeted = data_generator.flow(
    x_augumeted, y_augumeted,
    batch_size=augumet_size,
    shuffle=False
).next()[0]

#concatenate
x_train = np.concatenate((x_train, x_augumeted))
y_train = np.concatenate((y_train, y_augumeted))

print(x_train.shape, y_train.shape)

#scaler
x_train = x_train/255.
x_test = x_test/255.


#one hot
ohe = OneHotEncoder(sparse=False)
y_train = ohe.fit_transform(y_train)
y_test = ohe.fit_transform(y_test)


#모델 구성
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

#3. 컴파일 훈련
model.compile(loss= 'categorical_crossentropy', optimizer='adam', metrics=['acc'])
import time
startTime = time.time()
history = model.fit(x_train, y_train, epochs=1, batch_size=1000, verbose=2, validation_split=0.2, callbacks=[
    EarlyStopping(monitor='val_loss', mode='min', patience=30, restore_best_weights=True)])
endTime = time.time()

#모델저장
model.save("..\_data\_save\cifar10\keras31_cann5_save_model.h5") 

#4. 평가, 예측
# results = model.evaluate(x_test, y_test)
predict = ohe.inverse_transform(model.predict(x_test))
acc_score =  accuracy_score(ohe.inverse_transform(y_test), predict)
# print('loss = ', results[0])
# print('acc = ', results[1])
print('acc_score = ', acc_score)
print(predict)

'''
===============   증폭 전     =================
acc =  0.4444
===============30000개 증폭 후=================
acc_score : 0.8086
'''