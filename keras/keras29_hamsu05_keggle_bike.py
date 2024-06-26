import pandas as pd
import numpy as np
path = 'C:/_data/kaggle/bike/'
train_csv =pd.read_csv(path + 'train.csv', index_col=0)
test_csv =pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sampleSubmission.csv')

# train_csv.drop(75)
# test_csv.drop(75)


#데이터 구조 확인
print(train_csv.shape)#(10886, 11)
print(test_csv.shape)#(6493, 8)

# #결측치 확인
# print(train_csv.isna().sum())
# '''
# datetime      0
# season        0
# holiday       0
# workingday    0
# weather       0
# temp          0
# atemp         0
# humidity      0
# windspeed     0
# casual        0
# registered    0
# count         0
# '''
# print(test_csv.isna().sum())

# '''
# datetime      0
# season        0
# holiday       0
# workingday    0
# weather       0
# temp          0
# atemp         0
# humidity      0
# windspeed     0
# '''

#데이터 전처리
x = train_csv.drop('count', axis=1).drop('casual', axis=1).drop('registered', axis=1)
# x = train_csv.drop(columns='casual').drop(columns='registered').drop(columns= 'count')
print(x)

print(x.columns)
y = train_csv['count']


print(x.shape) #(10886, 8)
print(y.shape) #(10886, )

#####################*****B O A R D *****######################
train_size = 0.7
random_state = 12345
epochs = 1100
batch_size=10000
###############################################################

    
#데이터
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size= train_size, random_state= random_state)

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
# scaler = MinMaxScaler()
scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

#모델 구성
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input

input1 = Input(shape=(len(x.columns),))
dense1 = Dense(512)(input1)
dense2 = Dense(256)(dense1)
dense3 = Dense(128)(dense2)
dense4 = Dense(64)(dense3)
dense5 = Dense(32)(dense4)
dense6 = Dense(16)(dense5)
dense7 = Dense(8,activation='relu')(dense6)
dense8 = Dense(4,activation='relu')(dense7)
dense9 = Dense(2,activation='relu')(dense8)
output1 = Dense(1)(dense9)
model = Model(inputs = input1, outputs = output1)

#Early Stopping
from keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_accuracy', mode='max', patience= 1100, verbose=1, restore_best_weights=True)
import datetime
date = datetime.datetime.now()
print(date) #2024-01-17 10:52:41.770061
date = date.strftime("%m%d_%H%M")
print(date)


mcp_path = '../_data/_save/MCP/keggle/bike/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
filepath = "".join([mcp_path, 'k26_05_keggle_bike_' ,date, '_', filename]) #체크포인트 가장 좋은 결과들 저장
mcp = ModelCheckpoint(monitor='val_loss', mode='min',save_best_only=True, filepath=filepath, verbose=1 )

# 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae', 'acc'])
hist = model.fit(x_train, y_train, epochs= epochs, batch_size=batch_size, verbose=1, validation_split=0.3, callbacks=[es, mcp])

# 평가, 예측
loss = model.evaluate(x_test, y_test)


from sklearn.metrics import r2_score , mean_squared_error, mean_squared_log_error
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
submit = model.predict(test_csv)

# #데이터 출력
submission_csv['count'] = submit
nagativeCount = submission_csv[submission_csv['count']<0].count()
print( "음수의 개수: ", nagativeCount )

if nagativeCount['count'] == 0 :
    def RMSE(y_test, y_predict):
            return np.sqrt(mean_squared_error(y_test, y_predict)) 
    print('MSE : ', loss)
    rmse = RMSE(y_test, y_predict)
    print("RMSE : ", rmse)
    print('r2 = ', r2)

    print("="*100)
    
    import time as tm
    ltm = tm.localtime()
    file_name = f'sampleSubmission_{ltm.tm_year}{ltm.tm_mon}{ltm.tm_mday}{ltm.tm_hour}{ltm.tm_min}{ltm.tm_sec}.csv'
    submission_csv.to_csv(path + file_name, index = False )

    df_new = pd.DataFrame({'random_state' : [random_state], 'epoch' : [epochs], 'train_size' : [train_size], 
                        'batch_size' : [batch_size],'file_name' : [file_name],  'MSE' : [loss], 'RMSE': [rmse], 'r2': [r2]}) 
    df_new.to_csv(path + f"result_{ltm.tm_year}{ltm.tm_mon}{ltm.tm_mday}.csv", mode= 'a', header=True)
    #시각화
    import matplotlib.pyplot as plt
    hist_loss = hist.history['loss']
    hist_val_loss = hist.history['val_loss']

    plt.rcParams['font.family'] ='Malgun Gothic'
    plt.figure(figsize=(9,6))
    plt.plot(hist_loss, c ='red', label = 'loss', marker = '.')
    plt.plot(hist_val_loss, c = 'blue', label = 'val_loss', marker = '.')
    plt.legend(loc = 'upper right')
    plt.title('바이크 찻트')
    plt.xlabel = 'epoch'
    plt.ylabel = 'loss'
    plt.grid()
    plt.show()

    # def RMSLE(y_test, y_predict):
    #     return np.sqrt(mean_squared_log_error(y_test, y_predict))

    # rmsle = RMSLE(y_test, y_predict)
    # print('RMSLE : ', rmsle)
'''
기존 : 
loss :   [68176.6484375, 68176.6484375, 189.3523406982422, 0.010716472752392292]
============================
best : MaxAbs
============================
MinMaxScaler()
 - loss : [23940.2734375, 23940.2734375, 116.05174255371094, 0.010716472752392292]
StandardScaler()
 - loss :  [22986.853515625, 22986.853515625, 110.59112548828125, 0.010716472752392292]
MaxAbsScaler()
 - loss :  [68176.6484375, 68176.6484375, 189.35243225097656, 0.010716472752392292]
RobustScaler()
 - loss : [23383.75, 23383.75, 114.64836883544922, 0.010716472752392292]
 
Dropout 적용: 
MSE :  [23975.28515625, 23975.28515625, 116.01972961425781, 0.009797917678952217]
'''