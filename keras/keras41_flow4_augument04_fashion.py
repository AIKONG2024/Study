from keras.datasets import fashion_mnist
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from sklearn.metrics import accuracy_score
from keras.preprocessing.image import ImageDataGenerator

(x_train, x_test), (y_train, y_test) = fashion_mnist.load_data()
print(x_train.shape, y_train.shape)#(60000, 28, 28) (10000, 28, 28)
print(x_test.shape, y_test.shape)#(60000,) (10000,)

#증폭
data_generator =  ImageDataGenerator(
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=20,
    zoom_range=0.2
)

augumet_size = 40000

#rand
randidx = np.random.randint(x_train.shape[0], augumet_size)

data_generator.flow(
    
)