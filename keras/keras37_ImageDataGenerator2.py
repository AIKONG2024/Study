# x, y 추출해서 모델 만들기
# 성능 0.99 이상
#변환시간 체크
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,         
    horizontal_flip=True,    
    vertical_flip=True,      
    width_shift_range=0.1,   
    height_shift_range=0.1, 
    rotation_range=5,        
    zoom_range=1.2,         
    shear_range=0.7,      
    fill_mode='nearest',    
)

test_datagen = ImageDataGenerator(
    rescale=1./255        
)

path_train = 'c:/_data/image/brain/train/' 
path_test = 'c:/_data/image/brain/test/' 

xy_train =  train_datagen.flow_from_directory(
    path_train,
    target_size= (200,200),
    batch_size=160,        
    class_mode='binary',
    shuffle=True,
)

xy_test =  test_datagen.flow_from_directory(
    path_test,
    target_size= (200,200),
    batch_size=120,
    class_mode='binary',
)


#1. 데이터 
x_train = xy_train[0][0]
y_train = xy_train[0][1]

x_test = xy_test[0][0]
y_test = xy_test[0][1]

print(x_train.shape, y_train.shape) #(160, 200, 200, 3) (160,)
print(x_test.shape, y_test.shape) #(120, 200, 200, 3) (120,)

#scaling
x_train = x_train/255.
x_test = x_test/255.

#2. 모델 구성
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Dropout, MaxPooling2D, Flatten

model = Sequential()
model.add(Conv2D(16, (3,3), input_shape = (200,200,3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(Conv2D(32, (2,2), padding='same', activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(Conv2D(16, (2,2), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(Conv2D(256, (2,2), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

from keras.callbacks import EarlyStopping, ModelCheckpoint
# es = EarlyStopping(monitor='val_loss', mode='min', patience=100, restore_best_weights= True)

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=1000, batch_size=300, validation_split=0.2, callbacks=[])

#4.평가 예측

loss = model.evaluate(x_test, y_test)
predict = model.predict(x_test)

print('loss : ', loss[0])
print('acc : ', loss[1])