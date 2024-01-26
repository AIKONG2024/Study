from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPooling2D
from sklearn.metrics import accuracy_score
import numpy as np
import os
from sklearn.model_selection import train_test_split

#1. 데이터

#가져오기 && 세이브
train_data_generator = ImageDataGenerator()

test_data_generator = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=20,
    zoom_range=0.2
)

augumet_data_generator = ImageDataGenerator(
        horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=20,
    zoom_range=0.2
)

path_train ='C:\_data\image\horse_human/'
# path_test ='C:\_data\image\horse_human/'
image_path = 'C:\_data\image\horse_human/test/'
path = 'C:\_data/kaggle/horse_human/'
np_path = 'C:\_data\_save_npy\horse_human/'

train_x_npy_exists = os.path.exists(np_path + "keras_horse_human_x_train.npy") 
train_y_npy_exists = os.path.exists(np_path + "keras_horse_human_y_train.npy") 
# test_npy_exists = os.path.exists(np_path + "keras_horse_human_x_test.npy")

x_train = []
y_train = []
x_test = []

if not train_x_npy_exists and not train_y_npy_exists :#train npy 존재
    xy_train = train_data_generator.flow_from_directory(
    directory=path_train,
    batch_size=1027,
    target_size=(200,200),
    color_mode='rgb',
    class_mode='categorical',
    shuffle=True
    )
    x_train = xy_train[0][0]
    y_train = xy_train[0][1]
    np.save(np_path + 'keras_horse_human_x_train.npy', arr=x_train)
    np.save(np_path + 'keras_horse_human_y_train.npy', arr=y_train)
    
else: 
    x_train = np.load(np_path + 'keras_horse_human_x_train.npy')
    y_train = np.load(np_path + 'keras_horse_human_y_train.npy')
    
# if not test_npy_exists :#test npy 존재
#     xy_test = test_data_generator.flow_from_directory(
#     directory=path_test,
#     batch_size=5000,
#     target_size=(200,200),
#     color_mode='rgb',
#     class_mode='binary',
#     shuffle=True
#     )    
#     x_test = xy_test[0][0]
#     np.save(np_path + 'keras_horse_human_x_test.npy', arr=x_test)
# else:
#     x_test = np.load(np_path + 'keras_horse_human_x_test.npy')
    

#증폭
augumet_size = 3000

randidx = np.random.randint(x_train.shape[0],size = augumet_size)

x_augumeted = x_train[randidx].copy()
y_augumeted = y_train[randidx].copy()

x_augumeted = augumet_data_generator.flow(
    x_augumeted, y_augumeted,
    batch_size=augumet_size,
    shuffle=True
).next()[0]

x_train = np.concatenate((x_augumeted, x_train))
y_train = np.concatenate((y_augumeted, y_train))

print(x_train.shape)#(11027, 200, 200, 3)


x_train, x_test, y_train, y_test =  train_test_split(x_train, y_train, train_size=0.8, shuffle=True, random_state=777)

#2. 모델 구성
model = Sequential()

model.add(Conv2D(10, (2,2), input_shape=(200,200,3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(Conv2D(16, (3,3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(Conv2D(20, (3,3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(2, activation='softmax'))

from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', mode = 'min', patience=100, restore_best_weights=True)

#3. 컴파일, 훈련
hist = model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs= 1000, batch_size= 100, validation_split= 0.2, callbacks=[es])

#4. 평가 예측
predict = np.round(model.predict(x_test))
print(predict)
y_test = np.round(y_test)
acc_score = accuracy_score(y_test, predict)
print('acc_score :', acc_score)

'''
===============   증폭 전     =================
acc :  1.0
===============10000개 증폭 후=================
acc : 0.570719602977667
'''