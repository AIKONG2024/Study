import pandas as pd
import numpy as np
import time as tm

path = "C:/_data/dacon/diabetes/"

#데이터 가져오기
train_csv =  pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "sample_submission.csv")
#데이터 확인
print(train_csv.columns)
print(test_csv.columns)

# print(train_csv.isna().sum())
# print(test_csv.isna().sum())

#결측 데이터 제거

p_index = train_csv[train_csv['Pregnancies'] == 0].index
train_csv.drop(p_index, inplace=True)
p_index = train_csv[train_csv['Glucose'] == 0].index
train_csv.drop(p_index, inplace=True)
p_index = train_csv[train_csv['BloodPressure'] == 0].index
train_csv.drop(p_index, inplace=True)
p_index = train_csv[train_csv['SkinThickness'] == 0].index
train_csv.drop(p_index, inplace=True)
p_index = train_csv[train_csv['Insulin'] == 0].index
train_csv.drop(p_index, inplace=True)
p_index = train_csv[train_csv['BMI'] == 0].index
train_csv.drop(p_index, inplace=True)

x = train_csv.drop(columns='Outcome')
y = train_csv['Outcome']


# ====================================
random_state = 123123
train_size = 0.70
# ====================================
#데이터
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_size, random_state= random_state)

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)
# test_csv = scaler.transform(test_csv)

#모델 구성
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input

input1 = Input(shape=(len(x.columns),))
dense1 = Dense(256)(input1)
drop1 = Dropout(0.2)(dense1)
dense2 = Dense(128)(drop1)
drop2 = Dropout(0.2)(dense2)
dense3 = Dense(64)(drop2)
drop3 = Dropout(0.3)(dense3)
dense4 = Dense(32)(drop3)
dense5 = Dense(16)(dense4)
output1 = Dense(1, activation='sigmoid')(dense5)
model = Model(inputs = input1, outputs = output1)

#===============================
epochs = 10000
batch_size = 1
validation_split = 0.2
patience = 0
restore_best_weights = True
#===============================



#컴파일, 훈련
from keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', mode='min', patience=patience,  restore_best_weights=restore_best_weights)
import datetime
date = datetime.datetime.now()
print(date) #2024-01-17 10:52:41.770061
date = date.strftime("%m%d_%H%M")
print(date)


mcp_path = '../_data/_save/MCP/dacon/diabetes/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
filepath = "".join([mcp_path, 'k26_07_dacon_diabetes_' ,date, '_', filename]) #체크포인트 가장 좋은 결과들 저장
mcp = ModelCheckpoint(monitor= 'val_loss', mode = 'min', verbose=1 , save_best_only= True, filepath=filepath)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
start_time = tm.time()
history = model.fit(x_train, y_train, epochs=epochs, batch_size= batch_size, validation_split=validation_split, callbacks=[es, mcp])
end_time = tm.time()
#평가 예측
from sklearn.metrics import accuracy_score
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
accuracy = accuracy_score(y_test, np.round(y_predict))
submission = np.round(model.predict(test_csv))
print("loss : ", loss)
print("accuracy",accuracy )

#제출파일
submission_csv['Outcome'] = submission
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
                        'batch_size' : [batch_size],'file_name' : [file_path]}) 
df_new.to_csv(path + f"log_1_{ltm.tm_year}{ltm.tm_mon}{ltm.tm_mday}.csv", mode= 'a', header=True)

#시각화
import matplotlib.pyplot as plt
plt.figure(figsize=(9,4))
plt.plot(history_loss, color = 'red', label = "loss", marker = '.')
plt.plot(history_val_loss, color = 'blue', label = "val_loss", marker = '.')
plt.plot(history_accuracy, color = 'green', label = "accuracy", marker = '.')
plt.grid()
plt.show()



'''
기존 : 
 [0.5735149383544922, 0.7356321811676025]
============================
best : MaxAbs
============================
MinMaxScaler()
 - [0.6233892440795898, 0.7471264600753784]
StandardScaler()
 -  [0.6286416053771973, 0.7701149582862854]
MaxAbsScaler()
 -  [0.6220222115516663, 0.7471264600753784]
RobustScaler()
 -  [0.6357422471046448, 0.7586206793785095]
 
 Dropout() 적용후
 loss :  [1.6400569677352905, 0.5517241358757019]
accuracy 0.5517241379310345
'''