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

#결측치 확인
print(train_csv.isna().sum())
'''
datetime      0
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
'''
print(test_csv.isna().sum())

'''
datetime      0
season        0
holiday       0
workingday    0
weather       0
temp          0
atemp         0
humidity      0
windspeed     0
'''

#데이터 전처리
x = train_csv.drop('count', axis=1).drop('casual', axis=1).drop('registered', axis=1)
# x = train_csv.drop(columns='casual').drop(columns='registered').drop(columns= 'count')
print(x)

print(x.columns)
y = train_csv['count']


print(x.shape) #(10886, 8)
print(y.shape) #(10886, )

#####################*****B O A R D *****######################
train_size = 0.8
random_state = 123456743
epochs = 11000
batch_size=200
###############################################################

    
#데이터
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size= train_size, random_state= random_state)

#모델 구성
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(1024, input_dim = len(x.columns)))
model.add(Dense(512))
model.add(Dense(256))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
# model.add(Dense(2, activation='relu' ))
model.add(Dense(1))

#Early Stopping
from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', mode='min', patience= 300, verbose=1, restore_best_weights=True)

# 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
hist = model.fit(x_train, y_train, epochs= epochs, batch_size=batch_size, verbose=1, validation_split=0.3, callbacks=[es])

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
    