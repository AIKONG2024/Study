from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense,Dropout,Conv2D, MaxPooling2D, Flatten
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import OneHotEncoder

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

augumet_size = 50000

randidx = np.random.randint(x_train.shape[0], size= augumet_size)

x_augumeted = x_train[randidx].copy()
y_augumeted = y_train[randidx].copy()


data_generator.flow(
    x_augumeted,
    y_augumeted,
    batch_size=augumet_size,
    shuffle=True
)



