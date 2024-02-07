import pandas as pd
import numpy as np
path = 'C:/_data/kaggle/bike/'
train_csv =pd.read_csv(path + 'train.csv', index_col=0)
test_csv =pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sampleSubmission.csv')

#데이터 전처리
x = train_csv.drop('count', axis=1).drop('casual', axis=1).drop('registered', axis=1)
y = train_csv['count']
    
#데이터
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size= 0.7, random_state= 12345)

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

#모델 구성
from sklearn.svm import LinearSVR
model = LinearSVR(C=100)

# 컴파일, 훈련
model.fit(x_train, y_train)

# 평가, 예측
r2 = model.score(x_test, y_test)
y_predict = model.predict(x_test)
submit = model.predict(test_csv)

# #데이터 출력
submission_csv['count'] = submit
import time as tm
ltm = tm.localtime()
file_name = f'sampleSubmission_{ltm.tm_year}{ltm.tm_mon}{ltm.tm_mday}{ltm.tm_hour}{ltm.tm_min}{ltm.tm_sec}.csv'
submission_csv.to_csv(path + file_name, index = False )

print("r2 : ", r2)

'''
r2 :  0.22603070576013562
'''