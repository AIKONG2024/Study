from keras.datasets import reuters
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Conv1D, Dense, Embedding, LSTM, Flatten, MaxPooling1D
from keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score

directory_length = 21 #사전 단어의 개수+1
maxlen = 100

(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=directory_length)
# print("word index : ", word_index)
# print(x_train.shape, x_test.shape)#(8982,) (2246,)
# print(y_train.shape, y_test.shape)#(8982,) (2246,)
# print(type(x_train))#<class 'numpy.ndarray'>
# print(y_train) #[ 3  4  3 ... 25  3 25]
# unique, count = np.unique(y_train, return_counts=True)
# print(unique, count, len(unique))
# print(type(x_train[0]))#<class 'list'>
# print(len(x_train[0]))
# print(len(x_train[1]))
print(x_train)

print("뉴스기사의 최대길이 : ", max(len(i) for i in x_train)) #2376
print("뉴스기사의 평균길이 : ", sum(map(len, x_train))/len(x_train)) #145.5398574927633

#전처리
from keras.utils import pad_sequences
x_train = pad_sequences(x_train, padding='pre', maxlen=maxlen, 
                        truncating='pre')

x_test = pad_sequences(x_test, padding='pre', maxlen=maxlen, 
                        truncating='pre')

print(x_train.shape, y_train.shape) #(8982, 100) (8982,)

# 2.모델 구성

model = Sequential()
model.add(Embedding(input_dim = directory_length, output_dim= 64, input_length=maxlen))
model.add(LSTM(units=32, return_sequences=True,activation='relu'))
model.add(Conv1D(filters=32, kernel_size=2, activation='relu'))
model.add(MaxPooling1D(2))

model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dense(26, activation='sofrmax'))

# 3.모델 컴파일, 훈련
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train,y_train, epochs=1, batch_size=1000, validation_split=0.2, callbacks=[
    EarlyStopping(monitor='val_loss', mode='min', patience=100, restore_best_weights=True)
])
# 4.예측, 평가
loss = model.evaluate(x_test, y_test)
x_predict = np.argmax(model.predict(x_test), axis=1)
print("loss : ", loss)
acc_score = accuracy_score(y_test, x_predict)
print("acc_score: ", acc_score)

'''
loss :  [1.8995658159255981, 0.5115761160850525]
'''