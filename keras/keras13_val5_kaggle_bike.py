
import pandas as pd

path = 'C:/_data/kaggle/bike/'
train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + 'sampleSubmission.csv')

x = train_csv.drop('count', axis=1)
y = train_csv['count']

# 결측치확인
print(train_csv.isna().sum())
print(test_csv.isna().sum())
'''
season        0
holiday       0
workingday    0
weather       0
temp          0
atemp         0
humidity      0
windspeed     0
casual        0
registered    0
count         0
dtype: int64
season        0
holiday       0
workingday    0
weather       0
temp          0
atemp         0
humidity      0
windspeed     0
'''
print(train_csv.columns)
print(test_csv.columns)
'''
Index(['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp',
       'humidity', 'windspeed', 'casual', 'registered', 'count'],
      dtype='object')
Index(['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp',
       'humidity', 'windspeed'],
      dtype='object')
'''

#train casual registered 제거
x.drop(columns='casual', inplace= True)
x.drop(columns='registered', inplace= True)
# #데이터 분석
# print(x_train.shape)#(6531, 8)
# print(y_train.shape)#(6531,)
# print(x_test.shape)#(4355, 8)
# print(y_test.shape)#(4355,)


#####################*****B O A R D *****######################
train_size = 0.85
random_state = 9323
epochs = 500
batch_size= 300
validation_split = 0.3
###############################################################
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_size, random_state=random_state)

#모델 구성
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(64, input_dim = len(x.columns)))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))

#컴파일 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=epochs, batch_size= batch_size, verbose=3, validation_split=validation_split)

#평가 예측
loss = model.evaluate(x_test, y_test)

y_predict = model.predict(x_test)
submission = model.predict(test_csv)

from sklearn.metrics import r2_score, mean_squared_error, mean_squared_log_error
import numpy as np

r2_score = r2_score(y_test, y_predict)
rmse = np.sqrt(loss)
# rmsle = np.sqrt(mean_squared_log_error(y_test, y_predict))

print('mse : ', loss)
print('rmse :', rmse)
# print('rmsle :', rmsle)
# print('r2_score: ', r2_score)
nagativeCount = submission_csv[submission_csv['count']<0].count()
if  nagativeCount['count'] == 0 :
        #파일 출력
    submission_csv['count'] = submission
    import time as tm
    ltm = tm.localtime()
    file_name = f'sampleSubmission_{ltm.tm_year}{ltm.tm_mon}{ltm.tm_mday}{ltm.tm_hour}{ltm.tm_min}{ltm.tm_sec}.csv'
    submission_csv.to_csv(path + file_name, index = False )
    df_new = pd.DataFrame({'random_state' : [random_state], 'epoch' : [epochs], 'train_size' : [train_size], 
                    'batch_size' : [batch_size],'file_name' : [file_name],  'MSE' : [loss], 'RMSE': [rmse], 'r2': [r2_score]}) 
    df_new.to_csv(path + f"result_{ltm.tm_year}{ltm.tm_mon}{ltm.tm_mday}.csv", mode= 'a', header=True)
