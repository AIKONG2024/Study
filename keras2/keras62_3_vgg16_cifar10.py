import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.datasets import cifar10
from keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
import tensorflow as tf
import time
tf.random.set_seed(777)
np.random.seed(777)
print(tf.__version__) #2.9.0

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

from keras.applications import VGG16

vgg16 = VGG16(weights='imagenet', include_top=False, 
              input_shape=(32,32,3))
vgg16.trainable = False #가중치 동결 (훈련을 시키지 않음)

model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(256,activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.summary()

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
start_time = time.time()
model.fit(x_train, y_train, epochs=500, batch_size=5000, verbose=2, validation_split=0.2, callbacks=[
    EarlyStopping(monitor='val_loss', mode='min', patience=30, restore_best_weights=True)])
end_time = time.time()

predict = model.predict(x_test)
acc_score = accuracy_score(y_test, np.argmax(predict, axis=1))

print("acc score : ", acc_score)
print("time : ", round(end_time - start_time, 2))

'''
기존 acc_score : 0.8007
==============
VGG16 : 0.5472 
'''