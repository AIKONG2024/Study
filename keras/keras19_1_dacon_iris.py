#https://dacon.io/competitions/open/236070/
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

path = "C:/_data/dacon/iris/"

#1.데이터 가져오기
train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "sample_submission.csv")

#데이터 확인
print(train_csv.shape)#(120, 5)
print(test_csv.shape)#(30, 4)
print(submission_csv.shape)#(30, 2) "species"

x = train_csv.drop(columns='species')
y = train_csv['species']

#분류 클래스 확인
print(np.unique(y, return_counts=True)) #(array([0, 1, 2], dtype=int64), array([40, 41, 39], dtype=int64))

print(x.shape)#(120,4) 입력값: 4 출력값: 3
print(y.shape)#(120,)

#OneHotEncoder
one_hot_y = pd.get_dummies(y)

#데이터 분류
x_train, x_test, y_train, y_test = train_test_split(x, one_hot_y, train_size=0.7, random_state=200, stratify=one_hot_y)
print(x_test)
print(np.unique(y_test, return_counts=True))
#(array([False,  True]), array([72, 36], dtype=int64))

#모델 생성
model = Sequential()
model.add(Dense(64, input_dim = 4))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(3, activation='softmax'))

es = EarlyStopping(monitor='val_loss', mode = 'min', patience= 1000, restore_best_weights=True)

#컴파일 , 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=100000, batch_size=100, verbose= 1, validation_split=0.2, callbacks=[es])

#평가, 예측
loss = model.evaluate(x_test, y_test)
print("로스값 : ", loss)
y_predict = model.predict(x_test)
print(y_test, y_predict)
arg_y_test = np.argmax(y_test)
arg_y_predict = np.argmax(y_predict)
# print(arg_y_test, arg_y_predict)

# acc_score = accuracy_score(arg_y_test, arg_y_predict) 
print(model.predict(test_csv))
submission = np.argmax(model.predict(test_csv), axis=1) 
submission_csv['species'] = submission
import time as tm
ltm = tm.localtime(tm.time())
save_time = f"{ltm.tm_year}{ltm.tm_mon}{ltm.tm_mday}{ltm.tm_hour}{ltm.tm_min}{ltm.tm_sec}" 
file_path = path + f"sampleSubmission{save_time}.csv"
submission_csv.to_csv(file_path, index=False)
