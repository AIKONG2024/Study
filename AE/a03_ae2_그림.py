import numpy as np
from keras.datasets import mnist
import tensorflow as tf
import matplotlib.pyplot as plt

np.random.seed(0)
tf.random.set_seed(0)

#1. 데이터
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], 28*28).astype('float32')/255.
x_test = x_test.reshape(x_test.shape[0], 28*28).astype('float32')/255.

#평균 0, 표준편차 0.1인 정규분포
x_train_noised = x_train + np.random.normal(0, 0.3, size=x_train.shape)
x_test_noised = x_test + np.random.normal(0, 0.3, size=x_test.shape)

# x_train_noised[x_train_noised < 0] = 0
# x_train_noised[x_train_noised > 0] = 1
# x_test_noised[x_test_noised < 0] = 0
# x_test_noised[x_test_noised > 0] = 1
x_train_noised = np.clip(x_train_noised, 0, 1)
x_test_noised = np.clip(x_test_noised, 0, 1)

#2. 모델
from keras.models import Sequential, Model
from keras.layers import Dense, Input

def autoencoder(hidden_layer_size):
    model = Sequential()
    model.add(Dense(units= hidden_layer_size, input_shape=(28*28,)))
    model.add(Dense(784, activation='relu'))
    return model

'''
print(np.argmax(evr_cumsum >= 0.95) + 1) #154
print(np.argmax(evr_cumsum >= 0.99) + 1) #331
print(np.argmax(evr_cumsum >= 0.999) + 1) #486
print(np.argmax(evr_cumsum >= 1.0) + 1) #713
'''
hidden_size = 154
#best : 154 or 64

model = autoencoder(hidden_size)
model.summary()

#3. 컴파일, 훈련
model.compile(optimizer='adam', loss='mse')
# autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
# mse > binary_crossentropy

'''
best parameters: 
encoder activation : linear
encoder units : 64
decoder activation : sigmoid
loss : mse
'''

model.fit(x_train_noised, x_train, epochs=30, batch_size=128, validation_split=0.2)

#4. 평가, 예측
decoded_imgs = model.predict(x_test_noised)

from matplotlib import pyplot as plt
import random
fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10), (ax11, ax12, ax13, ax14, ax15)) = \
    plt.subplots(3,5, figsize=(20,7))

random_images = random.sample(range(decoded_imgs.shape[0]), 5)

#원본 이미지
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[random_images[i]].reshape(28,28), cmap='gray')
    if i == 0:
        ax.set_ylabel("INPUT", size=20)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])

#Noise 이미지
for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(x_test_noised[random_images[i]].reshape(28,28), cmap='gray')
    if i == 0:
        ax.set_ylabel("NOISE", size=20)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])

#Output 이미지
for i, ax in enumerate([ax11, ax12, ax13, ax14, ax15]):
    ax.imshow(decoded_imgs[random_images[i]].reshape(28,28), cmap='gray')
    if i == 0:
        ax.set_ylabel("OUTPUT", size=20)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])

plt.tight_layout()
plt.show()
