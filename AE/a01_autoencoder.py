import numpy as np
from keras.datasets import mnist

#1. 데이터
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], 28*28).astype('float32')/255.
x_test = x_test.reshape(x_test.shape[0], 28*28).astype('float32')/255.

#2. 모델
from keras.models import Sequential, Model
from keras.layers import Dense, Input

input_img = Input(shape=(784,))
##인코더
encoded = Dense(64, activation='linear')(input_img)
#linear > relu > sigmoid > tanh
# 64 > 128 > 32 > 1 == 1024

##디코더
decoded = Dense(784, activation='sigmoid')(encoded)
#sigmoid > relu > linear == tanh

autoencoder = Model(input_img, decoded)

autoencoder.summary()

#3. 컴파일, 훈련
autoencoder.compile(optimizer='adam', loss='mse')
# autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
# mse > binary_crossentropy

'''
best parameters: 
encoder activation : linear
encoder units : 64
decoder activation : sigmoid
loss : mse
'''

autoencoder.fit(x_train, x_train, epochs=30, batch_size=128, validation_split=0.2)

#4. 평가, 예측
decoded_imgs = autoencoder.predict(x_test)

import matplotlib.pyplot as plt
n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()