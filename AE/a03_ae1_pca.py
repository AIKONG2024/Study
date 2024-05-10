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

n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test_noised[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()