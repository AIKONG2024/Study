import pandas as pd

path = "C:/_data/dacon/cancer/"

#데이터 가져오기
train_csv =  pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "sample_submission.csv")
#데이터 확인
print(train_csv.columns)
print(test_csv.columns)

#결측치 확인 - 결측치 없음
# print(train_csv.isna().sum())
# print(test_csv.isna().sum())

x = train_csv.drop(columns='Outcome')
y = train_csv['Outcome']

# ====================================
random_state = 12345
train_size = 0.70
# ====================================
#데이터
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_size, random_state= random_state)

#모델 구성
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(64, input_dim = len(x.columns)))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(1, activation='sigmoid'))

#===============================
epochs = 1000000
batch_size = 10
validation_split = 0.2
patience = 100
restore_best_weights = True
#===============================


#컴파일, 훈련
from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_accuracy', mode='max', patience=patience,  restore_best_weights=restore_best_weights)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=epochs, batch_size= batch_size, validation_split=validation_split, callbacks=[es])

#평가 예측
from sklearn.metrics import accuracy_score
import numpy as np
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
accuracy = accuracy_score(y_test, np.round(y_predict))
submission = np.round(model.predict(test_csv))
print("loss : ", loss)
print("accuracy",accuracy )

#제출파일
submission_csv['Outcome'] = submission
import time as tm
ltm = tm.localtime(tm.time())
save_time = f"{ltm.tm_year}{ltm.tm_mon}{ltm.tm_mday}{ltm.tm_hour}{ltm.tm_min}{ltm.tm_sec}" 
file_path = path + f"sampleSubmission{save_time}.csv"
submission_csv.to_csv(file_path, index=False)
history_loss =  history.history['loss']
history_val_loss = history.history['val_loss']
history_val_acc = history.history['val_accuracy']
history_accuracy =  history.history['accuracy']

#기록
df_new = pd.DataFrame({'random_state' : [random_state], 'epoch' : [epochs], 'train_size' : [train_size], 
                        'batch_size' : [batch_size],'file_name' : [file_path],  'acc' : [accuracy], 'val_acc' : [history_val_acc]}) 
df_new.to_csv(path + f"log_1_{ltm.tm_year}{ltm.tm_mon}{ltm.tm_mday}.csv", mode= 'a', header=True)

#시각화
import matplotlib.pyplot as plt
plt.figure(figsize=(9,4))
plt.plot(history_loss, color = 'red', label = "loss", marker = '.')
plt.plot(history_val_loss, color = 'blue', label = "val_loss", marker = '.')
plt.plot(history_accuracy, color = 'green', label = "accuracy", marker = '.')
plt.grid()
plt.show()
