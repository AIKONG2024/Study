from keras.datasets import imdb
import numpy as np

dic_word_num = 10000
maxlen_size = 400
(x_train, y_train), (x_test, y_test) = imdb.load_data(
    num_words=dic_word_num,
    # maxlen=maxlen_size,
)

print(x_train.shape, y_train.shape) #(25000,) (25000,)
print(x_test.shape, y_test.shape) #(25000,) (25000,)
print(len(x_train[0]), len(x_train[0])) #218 218
print(y_train[:20])
#[1 0 0 1 0 0 1 0 1 0 1 0 0 0 0 0 1 1 0 1]
print(np.unique(y_train, return_counts= True))
#(array([0, 1], dtype=int64), array([12500, 12500], dtype=int64))

#####[실습] #####
from keras.utils import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, LSTM, Embedding, Flatten

x_train = pad_sequences(
    x_train,
    maxlen=maxlen_size,
    padding='pre'
)

x_test = pad_sequences(
    x_test,
    maxlen=maxlen_size,
    padding='pre'
)

# 2.모델구성
model = Sequential()
model.add(Embedding(input_dim = dic_word_num, output_dim= 64, input_length=maxlen_size))
# model.add(LSTM(units=64, return_sequences=True))
model.add(LSTM(units=32, return_sequences=True,activation='relu'))
model.add(Conv1D(filters=32, kernel_size=2, activation='relu'))
model.add(MaxPooling1D(2))

model.add(Flatten())
# model.add(Dense(32,activation='relu'))
# model.add(Dense(32,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(1, activation='sigmoid'))

from keras.callbacks import EarlyStopping
# 3. 모델 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=1000, batch_size=3000, validation_split=0.2, callbacks=[
    EarlyStopping(monitor='val_loss', mode='min', patience=100, restore_best_weights=True)
])

# 4. 평가, 예측
from sklearn.metrics import accuracy_score
loss = model.evaluate(x_test, y_test)

print('loss : ', loss[0])
print('acc : ', loss[1])

x_predict = np.round(model.predict(x_test))
print("predict :", x_predict)
acc_score = accuracy_score(y_test, x_predict)
print("acc_score :", acc_score)

# maxlen :100 acc_score : 0.8368
# maxlen :200 acc_score : 0.8618
# maxlen :300 acc_score : 0.86876
# maxlen :400 acc_score : 0.74444