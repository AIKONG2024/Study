# https://dacon.io/competitions/open/235576/mysubmission

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.utils import all_estimators
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')


#1. 데이터
path = "c:/_data/dacon/ddarung/"
train_csv = pd.read_csv(path + "train.csv", index_col=0) 
test_csv = pd.read_csv(path + "test.csv", index_col=0) 
submission_csv = pd.read_csv(path + "submission.csv")
train_csv = train_csv.fillna(test_csv.mean()) # 715 non-null
test_csv = test_csv.fillna(test_csv.mean()) # 715 non-null

x = train_csv.drop(['count'], axis=1) #axis 0이 행 1이 열
y = train_csv['count'] 

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size= 0.7, random_state=1234)
print(x_test.shape)

from sklearn.preprocessing import MinMaxScaler 
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_splits = 5 
kf = KFold(n_splits=n_splits, shuffle=True, random_state=123)

# 모델구성
model = HistGradientBoostingRegressor()

#3. 훈련
scores = cross_val_score(model, x, y, cv=kf)

# 평가, 예측
print("[스케일링 + train_test_split]")
print("r2 : ", scores, "\n평균 acc :", round(np.mean(scores),4))

y_predict = cross_val_predict(model, x_test, y_test, cv=kf)
# print(y_predict)
acc_score = r2_score(y_test, y_predict)
print("eval r2 :", acc_score)       

'''
============================================================
[The Best score] :  0.7850846854458855
[The Best model] :  HistGradientBoostingRegressor        
============================================================
[kfold 적용 후]
acc :  [0.77242699 0.7878237  0.76157915 0.80418396 0.79338383]
평균 acc : 0.7839
============================================================
[스케일링 + train_test_split]
r2 :  [0.77242699 0.7878237  0.76157915 0.80418396 0.79338383]
평균 acc : 0.7839
eval r2 : 0.7273867594196499
'''