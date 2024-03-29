from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPooling2D
from sklearn.metrics import accuracy_score
import numpy as np
import os

#1. 데이터

#가져오기 && 세이브
data_generator = ImageDataGenerator(
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=20,
    zoom_range=0.2
)

path_train ='c:/_data/image/cat_and_dog/train/'
path_test ='c:/_data/image/cat_and_dog/test/'
image_path = 'C:/_data/image/cat_and_dog/test/rand/'
path = 'C:/_data/kaggle/cat_and_dog/'
np_path = '../_data/_save_npy/'

train_x_npy_exists = os.path.exists(np_path + "keras_catdog_x_train.npy") 
train_y_npy_exists = os.path.exists(np_path + "keras_catdog_y_train.npy") 
test_npy_exists = os.path.exists(np_path + "keras_catdog_x_test.npy")

x_train = []
y_train = []
x_test = []

if not train_x_npy_exists and not train_y_npy_exists :#train npy 존재
    xy_train = data_generator.flow_from_directory(
    directory=path_train,
    batch_size=20000,
    target_size=(200,200),
    color_mode='rgb',
    class_mode='binary',
    shuffle=True
    )
    x_train = xy_train[0][0]
    y_train = xy_train[0][1]
    np.save(np_path + 'keras_catdog_x_train.npy', arr=x_train)
    np.save(np_path + 'keras_catdog_y_train.npy', arr=y_train)
    
else: 
    x_train = np.load(np_path + 'keras_catdog_x_train.npy')
    y_train = np.load(np_path + 'keras_catdog_y_train.npy')
    
if not test_npy_exists :#test npy 존재
    xy_test = data_generator.flow_from_directory(
    directory=path_test,
    batch_size=5000,
    target_size=(200,200),
    color_mode='rgb',
    class_mode='binary',
    shuffle=True
    )    
    x_test = xy_test[0][0]
    np.save(np_path + 'keras_catdog_x_test.npy', arr=x_test)
else:
    x_test = np.load(np_path + 'keras_catdog_x_test.npy')
    

#증폭
augumet_size = 10000

randidx = np.random.randint(x_train.shape[0],size = augumet_size)

x_augumeted = x_train[randidx].copy()
y_augumeted = y_train[randidx].copy()

x_augumeted = data_generator.flow(
    x_augumeted, y_augumeted,
    batch_size=augumet_size,
    shuffle=True
).next()[0]

x_train = np.concatenate((x_augumeted, x_train))

print(x_train.shape)#(29989, 200, 200, 3)


#2. 모델 구성
model = Sequential()

model.add(Conv2D(32, (2,2), input_shape=(200,200,3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))
s
from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', mode = 'min', patience=300, restore_best_weights=True)

#3. 컴파일, 훈련
hist = model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs= 1000, batch_size= 5, validation_split= 0.2, callbacks=[es])

#4. 평가 예측
predict = np.round(model.predict(x_test))
print(predict)

print('loss : ', hist.history[0])
print('acc : ', hist.history[1])

'''
===============   증폭 전     =================
loss :  0.46612799167633057
acc :  0.7822499871253967
===============10000개 증폭 후=================
acc
'''